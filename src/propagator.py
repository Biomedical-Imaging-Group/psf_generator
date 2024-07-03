import torch
import numpy as np
from abc import ABC, abstractmethod
from torch.special import bessel_j0, bessel_j1
from scipy.special import itj0y0
from functorch import vmap
from .utils.czt import custom_ifft2
from .integrators import trapezoid_rule, simpsons_rule, richard2_rule

# # re-enable if gradients wrt Bessel term are required
# from bessel_ad import BesselJ0
# bessel_j0_ad = BesselJ0.apply


class Propagator(ABC):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=1.3, fov=2000, refractive_index=1.5, 
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None, 
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3, 
                 n_i=1.5, t_i0=100e3):
        self.pupil = pupil
        
        self.n_pix_psf = n_pix_psf
        self.n_pix_pupil = pupil.n_pix_pupil
        self.device = device
        if self.device != pupil.device:
            print('Warning: device of propagator and pupil are not the same.')
            print('Pupil device: ', pupil.device)
            print('Propagator device: ', self.device)
            print('Setting propagator device to pupil device.')
            self.device = pupil.device

        # All distances are in nanometers
        self.wavelength = wavelength
        self.NA = NA
        self.fov = fov
        self.refractive_index = refractive_index

        self.defocus_min = defocus_min
        self.defocus_max = defocus_max
        self.n_defocus = n_defocus

        self.apod_factor = apod_factor
        self.envelope = envelope

        self.gibson_lanni = gibson_lanni
        self.z_p = z_p
        self.n_s = n_s
        self.n_g = n_g
        self.n_g0 = n_g0
        self.t_g = t_g
        self.t_g0 = t_g0
        self.n_i = n_i
        self.n_i0 = refractive_index
        self.t_i0 = t_i0
        self.t_i = n_i * (t_g0/n_g0 + t_i0/self.n_i0 - t_g/n_g - z_p/n_s)

        self.field = None

    @abstractmethod
    def compute_focus_field(self):
        raise NotImplementedError


class ScalarCartesianPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=0.9, fov=1000, refractive_index=1.5, 
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 sz_correction=True, apod_factor=False, envelope=None, 
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, t_i0=100e3):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, NA=NA, fov=fov, refractive_index=refractive_index, 
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope, 
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, t_i0=t_i0)
        self.sz_correction = sz_correction
        
         # Zoom factor to determine pixel size with custom FFT
        self.zoom_factor = 2 * self.NA * self.fov / self.wavelength \
             / self.refractive_index / (self.n_pix_pupil - 1)           # CHANGE: DIVIDE BY N-1 INSTEAD OF N

        # Compute coordinates s_x, s_y, s_z
        n_pix_pupil = self.pupil.n_pix_pupil
        s_x = torch.linspace(-1, 1, n_pix_pupil).to(self.device)
        s_xx, s_yy = torch.meshgrid(s_x, s_x, indexing='ij')
        s_zz = torch.sqrt((1 - (self.NA/self.refractive_index)**2 * (s_xx**2 + s_yy**2)
                          ).clamp(min=0.001)).reshape(1, 1, n_pix_pupil, n_pix_pupil)

        # Precompute additional factors
        self.k = 2 * np.pi / self.wavelength
        self.correction_factor = torch.ones(1, 1, n_pix_pupil, n_pix_pupil
                                            ).to(torch.complex64).to(self.device)
        if self.sz_correction:
            self.correction_factor *= 1 / s_zz
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(s_zz)
        if self.envelope is not None:
            self.correction_factor *= torch.exp(- (1-s_zz**2) / self.envelope**2)
        if self.gibson_lanni:
            # computed following Eq. (3.45) of François Aguet's thesis
            sin_t = (self.NA / self.refractive_index * torch.sqrt(s_xx**2 + s_yy**2)).clamp(max=1)
            optical_path = self.z_p * torch.sqrt(self.n_s**2 - self.n_i**2 * sin_t**2) \
                            + self.t_i * torch.sqrt(self.n_i**2 - self.n_i**2 * sin_t**2) \
                            - self.t_i0 * torch.sqrt(self.n_i0**2 - self.n_i**2 * sin_t**2) \
                            + self.t_g * torch.sqrt(self.n_g**2 - self.n_i**2 * sin_t**2) \
                            - self.t_g0 * torch.sqrt(self.n_g0**2 - self.n_i**2 * sin_t**2)
            self.correction_factor *= torch.exp(1j * self.k * optical_path)
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus
                                       ).reshape(-1, 1, 1, 1).to(self.device)
        self.defocus_filters = torch.exp(1j * self.k * s_zz * defocus_range)

        self.s_x = s_x
        self.s_max = torch.tensor(self.NA / self.refractive_index)

        ds = s_x[1] - s_x[0]
        self.ds = ds
        
        k_range_fft = 1.0 / ds
        k_start = -self.zoom_factor * np.pi
        k_end   =  self.zoom_factor * np.pi
        x = torch.linspace(k_start, k_end, self.n_pix_pupil) / (2.0 * torch.pi) * k_range_fft
        self.x = x


    def compute_focus_field(self):
        self.field = custom_ifft2(self.pupil.field * self.correction_factor * self.defocus_filters, 
                                  shape_out=(self.n_pix_psf, self.n_pix_psf), 
                                  k_start=-self.zoom_factor*np.pi, 
                                  k_end=self.zoom_factor*np.pi, 
                                  norm='backward', fftshift_input=True, include_end=True) * \
                                    (2 * self.NA / self.n_pix_pupil)**2
        return self.field / (2 * np.pi)


    def _compute_PSF_for_far_field(self, far_fields):
        self.field = custom_ifft2(far_fields * self.correction_factor * self.defocus_filters, 
                                  shape_out=(self.n_pix_psf, self.n_pix_psf), 
                                  k_start  = -self.zoom_factor * np.pi, 
                                  k_end    =  self.zoom_factor * np.pi, 
                                  norm='backward', 
                                  fftshift_input=True, 
                                  include_end=True) \
                                      * (self.ds * self.s_max) ** 2 * 1j    # CHANGE: different scale factor and 1j phase
        
        return self.field / (2 * np.pi)


class ScalarPolarPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=0.9, fov=1000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None,  
                 gibson_lanni=False, z_p=1e3, n_s=1.3, 
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, t_i0=100e3, 
                 quadrature_rule=simpsons_rule):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, NA=NA, fov=fov, refractive_index=refractive_index,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus, 
                         apod_factor=apod_factor, envelope=envelope, 
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, t_i0=t_i0)
        
        # PSF coordinates
        x = torch.linspace(-self.fov/2, self.fov/2, self.n_pix_psf)
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        rr = torch.sqrt(xx ** 2 + yy ** 2)
        r_unique, rr_indices = torch.unique(rr, return_inverse=True)
        self.rs = r_unique.to(self.device)  # compute minimal number of points
        self.rr_indices = rr_indices.to(self.device)  # to invert

        # Pupil coordinates
        theta_max = np.arcsin(self.NA / self.refractive_index)
        self.thetas = torch.linspace(0, theta_max, self.n_pix_pupil).to(self.device)
        self.dtheta = theta_max / (self.n_pix_pupil - 1)

        # Precompute additional factors
        self.k = 2.0 * np.pi / self.wavelength
        sin_t, cos_t = torch.sin(self.thetas), torch.cos(self.thetas)
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus)
        self.defocus_filters = torch.exp(1j * self.k * defocus_range[:,None] * cos_t[None,:]).to(self.device)   # [n_defocus, n_thetas]

        correction_factor = torch.ones(self.n_pix_pupil)
        if self.apod_factor:
            correction_factor *= torch.sqrt(cos_t)
        if self.envelope is not None:
            correction_factor *= torch.exp(-(sin_t / self.envelope) ** 2)
        if self.gibson_lanni:
            # computed following Eq. (3.45) of François Aguet's thesis
            optical_path = self.z_p * torch.sqrt(self.n_s**2 - self.n_i**2 * self.sin_t**2) \
                            + self.t_i * torch.sqrt(self.n_i**2 - self.n_i**2 * self.sin_t**2) \
                            - self.t_i0 * torch.sqrt(self.n_i0**2 - self.n_i**2 * self.sin_t**2) \
                            + self.t_g * torch.sqrt(self.n_g**2 - self.n_i**2 * self.sin_t**2) \
                            - self.t_g0 * torch.sqrt(self.n_g0**2 - self.n_i**2 * self.sin_t**2)
            correction_factor *= torch.exp(1j * self.k * optical_path)
        self.correction_factor = correction_factor.to(self.device)
        self.quadrature_rule = quadrature_rule
        
        # bessel function evaluations are expensive and can be computed independently from defocus
        self.J_evals = bessel_j0(self.k * self.rs[None,:] * sin_t[:,None])    # [n_theta, n_radii]


    def compute_focus_field(self):
        far_fields = self.pupil.field.squeeze()   # [n_defocus=1, channels=1, n_thetas] ==> [n_thetas, ]
        return self._compute_PSF_for_far_field(far_fields)

    def _compute_PSF_for_far_field(self, far_fields):
        # argument shapes:
        # self.thetas,            [n_thetas, ]
        # self.dtheta,            float
        # self.rs,                [n_radii, ]
        # self.correction_factor  [n_thetas, ]
        # far_fields              [n_thetas, ]

        sin_t = torch.sin(self.thetas) # [n_thetas, ]

        # bessel function evaluations are expensive and can be computed independently from defocus
        J_evals = bessel_j0(self.k * self.rs[None,:] * sin_t[:,None])    # [n_theta, n_radii]

        # compute PSF field; handle defocus via batching with vmap()
        batched_compute_field_at_defocus = vmap(self._compute_PSF_at_defocus, in_dims=(0, None, None, None))

        fields = batched_compute_field_at_defocus(self.defocus_filters, J_evals, far_fields, sin_t)
        return fields        

    def _compute_PSF_at_defocus(self, defocus_term, J_evals, far_fields, sin_t):
        # compute E(r) for a list of unique radii values
        integrand = J_evals * (far_fields * defocus_term * self.correction_factor * sin_t)[:,None]  # [n_theta, n_radii]
        field = self.quadrature_rule(integrand, self.dtheta)
        # scatter the radial evaluations of E(r) onto the xy image grid
        field = field[self.rr_indices].unsqueeze(0)       # [n_channels=1, size_x, size_y]
        return field


class VectorialPolarPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=0.9, fov=1000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, n_i0=1.5, t_i0=100e3):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, NA=NA, fov=fov, refractive_index=refractive_index,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, t_i0=t_i0)

        # PSF coordinates
        x = torch.linspace(-self.fov / 2, self.fov / 2, self.n_pix_psf)
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        self.r = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(self.device)
        self.varphi = torch.atan2(yy, xx)

        # Pupil coordinates
        # Thetas are defined twice here, to be cleaned when working on VectPolarProp
        self.theta_max = np.arcsin(self.NA / self.refractive_index)
        theta = torch.linspace(0, self.theta_max, self.n_pix_pupil).to(self.device)
        # TODO: number of pixels in pupil (== gridsize of integration domain) should be driven
        # by the integration method and its required accuracy, not set a-priori
        # TODO: ideally, pupil should be described by a *continuous function* that allows us to query its
        # value at any value of `theta`
        theta_max = np.arcsin(self.NA / self.refractive_index)
        num_thetas = self.n_pix_pupil
        thetas = torch.linspace(0, theta_max, num_thetas)
        dtheta = theta_max / (num_thetas - 1)
        self.thetas = thetas.to(self.device)
        self.dtheta = dtheta

        # Precompute additional factors
        self.k = 2 * np.pi / self.wavelength
        self.sin_t = torch.reshape(torch.sin(theta), (1, 1, 1, 1, -1))
        self.cos_t = torch.reshape(torch.cos(theta), (1, 1, 1, 1, -1))
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus
                                       ).reshape(-1, 1, 1, 1, 1).to(self.device)
        self.defocus_filters = torch.exp(1j * self.k * defocus_range * self.cos_t)
        correction_factor = torch.ones(1, 1, 1, 1, self.n_pix_pupil).to(torch.complex64)
        if self.apod_factor:
            correction_factor *= torch.sqrt(self.cos_t)
        if self.envelope is not None:
            correction_factor *= torch.exp(- self.sin_t ** 2 / self.envelope ** 2)
        if self.gibson_lanni:
            # computed following Eq. (3.45) of François Aguet's thesis
            optical_path = self.z_p * torch.sqrt(self.n_s ** 2 - self.n_i ** 2 * self.sin_t ** 2) \
                           + self.t_i * torch.sqrt(self.n_i ** 2 - self.n_i ** 2 * self.sin_t ** 2) \
                           - self.t_i0 * torch.sqrt(self.n_i0 ** 2 - self.n_i ** 2 * self.sin_t ** 2) \
                           + self.t_g * torch.sqrt(self.n_g ** 2 - self.n_i ** 2 * self.sin_t ** 2) \
                           - self.t_g0 * torch.sqrt(self.n_g0 ** 2 - self.n_i ** 2 * self.sin_t ** 2)
            correction_factor *= torch.exp(1j * self.k * optical_path)
        self.correction_factor = correction_factor.to(self.device)

    def compute_focus_field(self):
        # multiplicative scalar factor to be verified for the vectorial case

        i0 = torch.sum(
            self.pupil.field.unsqueeze(-2).unsqueeze(-2) *
            bessel_j0(self.k * self.r * self.sin_t) *
            self.sin_t * (self.cos_t + 1) * self.defocus_filters
            , dim=-1) * self.theta_max / self.n_pix_pupil / np.sqrt(self.refractive_index)
        i1 = torch.sum(
            self.pupil.field.unsqueeze(-2).unsqueeze(-2) *
            bessel_j1(self.k * self.r * self.sin_t) *
            self.sin_t ** 2 * self.defocus_filters
            , dim=-1) * self.theta_max / self.n_pix_pupil / np.sqrt(self.refractive_index)
        i2 = torch.sum(
            self.pupil.field.unsqueeze(-2).unsqueeze(-2) *
            self.bessel_j2(self.k * self.r * self.sin_t) *
            self.sin_t * (self.cos_t - 1) * self.defocus_filters
            , dim=-1) * self.theta_max / self.n_pix_pupil / np.sqrt(self.refractive_index)

        self.field = torch.stack((i0[:, 0, :, :] + i2[:, 0, :, :] * torch.cos(2 * self.varphi) + i2[:, 1, :, :] * torch.sin(2 * self.varphi),
                                 i2[:, 0, :, :] * torch.sin(2 * self.varphi) + i0[:, 1, :, :] - i2[:, 1, :, :] * torch.cos(2 * self.varphi),
                                 -2 * 1j * i1[:, 0, :, :] * torch.cos(self.varphi) - 2 * 1j * i1[:, 1, :, :] * torch.sin(self.varphi)), dim=1)

        return self.field

    @staticmethod
    def bessel_j2(r):
        eps = 1e-10
        return 2 * bessel_j1(r) / (r+eps) - bessel_j0(r)

# The following is wrong -- needs to be debugged
class VectorialCartesianPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=0.9, fov=1000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 sz_correction=True, apod_factor=False, envelope=None,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, n_i0=1.5, t_i0=100e3):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, NA=NA, fov=fov, refractive_index=refractive_index,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, t_i0=t_i0)
        self.sz_correction = sz_correction

        # Zoom factor to determine pixel size with custom FFT
        self.zoom_factor = 2 * self.NA * self.fov / self.wavelength \
                           / self.refractive_index / self.n_pix_pupil

        # Compute coordinates s_x, s_y, s_z
        n_pix_pupil = self.pupil.n_pix_pupil
        x = torch.linspace(-1, 1, n_pix_pupil).to(self.device)
        s_x, s_y = torch.meshgrid(x, x, indexing='ij')
        s_xy2 = (s_x ** 2 + s_y ** 2).reshape(1, 1, n_pix_pupil, n_pix_pupil)
        s_z = torch.sqrt((1 - (self.NA / self.refractive_index) ** 2 * s_xy2
                          ).clamp(min=0.001)).reshape(1, 1, n_pix_pupil, n_pix_pupil)

        # Precompute additional factors
        self.k = 2 * np.pi / self.wavelength
        cos_theta = torch.sqrt(1 - s_xy2)
        sin_theta = torch.sqrt(s_xy2)
        cos_2phi = (s_x**2 - s_y**2) / s_xy2
        sin_2phi = 2 * s_x * s_y / s_xy2
        e_inc = self.pupil.field[0, 0].reshape(1, 1, n_pix_pupil, n_pix_pupil)
        far_field_x = ((cos_theta + 1) + (cos_theta-1) * cos_2phi) * e_inc
        self.far_field = 0.5 * far_field_x  # for the moment only x component for e_inc x-polarized computed
        self.correction_factor = torch.ones(1, 1, n_pix_pupil, n_pix_pupil
                                            ).to(torch.complex64).to(self.device)
        if self.sz_correction:
            self.correction_factor *= 1 / s_z    # s_z or cos_theta ? -- change to True when making sure!
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(s_z)   # s_z or cos_theta ?
        if self.envelope is not None:
            self.correction_factor *= torch.exp(- (1 - s_z ** 2) / self.envelope ** 2)
        if self.gibson_lanni:
            # computed following Eq. (3.45) of François Aguet's thesis
            sin_t = (self.NA / self.refractive_index * torch.sqrt(s_x ** 2 + s_y ** 2)).clamp(max=1)
            optical_path = self.z_p * torch.sqrt(self.n_s ** 2 - self.n_i ** 2 * sin_t ** 2) \
                           + self.t_i * torch.sqrt(self.n_i ** 2 - self.n_i ** 2 * sin_t ** 2) \
                           - self.t_i0 * torch.sqrt(self.n_i0 ** 2 - self.n_i ** 2 * sin_t ** 2) \
                           + self.t_g * torch.sqrt(self.n_g ** 2 - self.n_i ** 2 * sin_t ** 2) \
                           - self.t_g0 * torch.sqrt(self.n_g0 ** 2 - self.n_i ** 2 * sin_t ** 2)
            self.correction_factor *= torch.exp(1j * self.k * optical_path)
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus
                                       ).reshape(-1, 1, 1, 1).to(self.device)
        self.defocus_filters = torch.exp(1j * self.k * s_z * defocus_range)

    def compute_focus_field(self):
        self.field = custom_ifft2(self.far_field * self.correction_factor * self.defocus_filters,
                                  shape_out=(self.n_pix_psf, self.n_pix_psf),
                                  k_start=-self.zoom_factor * np.pi,
                                  k_end=self.zoom_factor * np.pi,
                                  norm='backward', fftshift_input=True, include_end=True) * \
                     (2 * self.NA / self.n_pix_pupil) ** 2
        return self.field / (2 * np.pi)
