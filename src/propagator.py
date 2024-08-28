from abc import ABC, abstractmethod

import numpy as np
import torch
from functorch import vmap
from torch.special import bessel_j0, bessel_j1

from integrators import simpsons_rule
from utils.czt import custom_ifft2

# from bessel_ad import bessel_j2
# # re-enable if gradients wrt Bessel term are required
# from bessel_ad import BesselJ0
# bessel_j0_ad = BesselJ0.apply

# Todo:
# - refractive_index and n_i are the same thing
# - check whether it's possible to remove torch.tensor in self.s_max in ScalarCartesianPropagator
# - integral normalization in ScalarCartesianPropagator


class Propagator(ABC):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, na=1.3, fov=2000, refractive_index=1.5,
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
        self.na = na
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

    def compute_optical_path(self, sin_t: torch.Tensor) -> torch.Tensor:
        """Computed following Eq. (3.45) of François Aguet's thesis"""
        path = self.z_p * torch.sqrt(self.n_s ** 2 - self.n_i ** 2 * sin_t ** 2) \
            + self.t_i * torch.sqrt(self.n_i ** 2 - self.n_i ** 2 * sin_t ** 2) \
            - self.t_i0 * torch.sqrt(self.n_i0 ** 2 - self.n_i ** 2 * sin_t ** 2) \
            + self.t_g * torch.sqrt(self.n_g ** 2 - self.n_i ** 2 * sin_t ** 2) \
            - self.t_g0 * torch.sqrt(self.n_g0 ** 2 - self.n_i ** 2 * sin_t ** 2)
        return path


class ScalarCartesianPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, na=1.3, fov=1000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 sz_correction=True, apod_factor=False, envelope=None,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, t_i0=100e3):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, na=na, fov=fov, refractive_index=refractive_index,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, t_i0=t_i0)
        self.sz_correction = sz_correction

        # Physical parameters
        self.k = 2 * np.pi / self.wavelength
        self.s_max = torch.tensor(self.na / self.refractive_index)

         # Zoom factor to determine pixel size with custom FFT
        self.zoom_factor = 2 * self.s_max * self.fov / self.wavelength \
             / (self.n_pix_pupil - 1)

        # Coordinates in pupil space s_x, s_y, s_z
        n_pix_pupil = self.pupil.n_pix_pupil
        self.s_x = torch.linspace(-1, 1, n_pix_pupil).to(self.device)
        self.ds = self.s_x[1] - self.s_x[0]
        s_xx, s_yy = torch.meshgrid(self.s_x, self.s_x, indexing='ij')
        s_zz = torch.sqrt((1 - self.s_max ** 2 * (s_xx ** 2 + s_yy ** 2)
                          ).clamp(min=0.001)).reshape(1, 1, n_pix_pupil, n_pix_pupil)

        # Coordinates in object space
        total_fft_range = 1.0 / self.ds
        k_start = -self.zoom_factor * np.pi
        k_end = self.zoom_factor * np.pi
        self.x = torch.linspace(k_start, k_end, self.n_pix_pupil) / (2.0 * torch.pi) * total_fft_range

        # Correction factors
        self.correction_factor = torch.ones(1, 1, n_pix_pupil, n_pix_pupil).to(torch.complex64).to(self.device)
        if self.sz_correction:
            self.correction_factor *= 1 / s_zz
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(s_zz)
        if self.envelope is not None:
            self.correction_factor *= torch.exp(- (1 - s_zz ** 2) / self.envelope ** 2)
        if self.gibson_lanni:
            clamp_value = np.minimum(self.n_s/self.n_i, self.n_g/self.n_i)
            sin_t = (self.s_max * torch.sqrt(s_xx**2 + s_yy**2)).clamp(max=clamp_value)
            path = self.compute_optical_path(sin_t)
            self.correction_factor *= torch.exp(1j * self.k * path)

        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus
                                       ).reshape(-1, 1, 1, 1).to(self.device)
        self.defocus_filters = torch.exp(1j * self.k * s_zz * defocus_range)

    def compute_focus_field(self):
        self.field = self._compute_psf_for_far_field(self.pupil.field)
        return self.field

    def _compute_psf_for_far_field(self, far_fields):
        k_start = -self.zoom_factor * np.pi
        k_end   =  self.zoom_factor * np.pi
        field = custom_ifft2(far_fields * self.correction_factor * self.defocus_filters,
                                  shape_out=(self.n_pix_psf, self.n_pix_psf),
                                  k_start=k_start,
                                  k_end=k_end,
                                  norm='forward', fftshift_input=True, include_end=True) \
                                      * (self.ds * self.s_max) ** 2
        return field / (2 * np.pi * np.sqrt(self.refractive_index))


class ScalarPolarPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, na=1.3, fov=1000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None, cos_factor=False,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, t_i0=100e3,
                 quadrature_rule=simpsons_rule):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, na=na, fov=fov, refractive_index=refractive_index,
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
        theta_max = np.arcsin(self.na / self.refractive_index)
        thetas = torch.linspace(0, theta_max, self.n_pix_pupil)
        self.thetas = thetas.to(self.device)
        self.dtheta = theta_max / (self.n_pix_pupil - 1)

        # Precompute additional factors
        self.cos_factor = cos_factor
        self.k = 2.0 * np.pi / self.wavelength
        sin_t, cos_t = torch.sin(thetas), torch.cos(thetas)
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus)
        self.defocus_filters = torch.exp(1j * self.k * defocus_range[:,None] * cos_t[None,:]).to(self.device)   # [n_defocus, n_thetas]

        self.correction_factor = torch.ones(self.n_pix_pupil).to(torch.complex64).to(self.device)
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(cos_t)
        if self.envelope is not None:
            self.correction_factor *= torch.exp(-(sin_t / self.envelope) ** 2)
        if self.gibson_lanni:
            clamp_value = np.minimum(self.n_s/self.n_i, self.n_g/self.n_i)
            sin_t = sin_t.clamp(max=clamp_value)
            path = self.compute_optical_path(sin_t)
            self.correction_factor *= torch.exp(1j * self.k * path)
        elif self.cos_factor:
            self.correction_factor *= cos_t

        self.quadrature_rule = quadrature_rule

        # bessel function evaluations are expensive and can be computed independently from defocus
        self.J_evals = bessel_j0(self.k * self.rs[None,:] * sin_t[:,None])    # [n_theta, n_radii]

    def compute_focus_field(self):
        far_fields = self.pupil.field.squeeze()   # [n_defocus=1, channels=1, n_thetas] ==> [n_thetas, ]
        return self._compute_psf_for_far_field(far_fields)

    def _compute_psf_for_far_field(self, far_fields):
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
        batched_compute_field_at_defocus = vmap(self._compute_psf_at_defocus, in_dims=(0, None, None, None))

        fields = batched_compute_field_at_defocus(self.defocus_filters, J_evals, far_fields, sin_t)
        return fields

    def _compute_psf_at_defocus(self, defocus_term, J_evals, far_fields, sin_t):
        # compute E(r) for a list of unique radii values
        integrand = J_evals * (far_fields * defocus_term * self.correction_factor * sin_t)[:,None]  # [n_theta, n_radii]
        field = self.quadrature_rule(integrand, self.dtheta)
        # scatter the radial evaluations of E(r) onto the xy image grid
        field = field[self.rr_indices].unsqueeze(0)       # [n_channels=1, size_x, size_y]
        return field / np.sqrt(self.refractive_index)


class VectorialCartesianPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, na=1.3, fov=1000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 sz_correction=True, apod_factor=False, envelope=None,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, t_i0=100e3):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, na=na, fov=fov, refractive_index=refractive_index,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, t_i0=t_i0)
        self.sz_correction = sz_correction

        # Physical parameters
        self.k = 2 * np.pi / self.wavelength
        self.s_max = torch.tensor(self.na / self.refractive_index)

        # Zoom factor to determine pixel size with custom FFT
        self.zoom_factor = 2 * self.s_max * self.fov / self.wavelength \
                            / (self.n_pix_pupil - 1)

        # Coordinates in pupil space s_x, s_y, s_z
        n_pix_pupil = self.pupil.n_pix_pupil
        self.s_x = torch.linspace(-1, 1, n_pix_pupil).to(self.device)
        self.ds = self.s_x[1] - self.s_x[0]
        s_xx, s_yy = torch.meshgrid(self.s_x, self.s_x, indexing='ij')
        s_zz = torch.sqrt((1 - self.s_max ** 2 * (s_xx ** 2 + s_yy ** 2)
                           ).clamp(min=0.001)).reshape(1, 1, n_pix_pupil, n_pix_pupil)

        # Coordinates in object space
        total_fft_range = 1.0 / self.ds
        k_start = -self.zoom_factor * np.pi
        k_end = self.zoom_factor * np.pi
        self.x = torch.linspace(k_start, k_end, self.n_pix_pupil) / (2.0 * torch.pi) * total_fft_range

        # Angles theta and phi
        sin_xx, sin_yy = torch.meshgrid(self.s_x * self.s_max, self.s_x * self.s_max, indexing='ij')
        sin_t_sq = sin_xx ** 2 + sin_yy ** 2
        s_valid = sin_t_sq <= self.s_max ** 2
        sin_theta = torch.sqrt(sin_t_sq)
        cos_theta = torch.sqrt(1.0 - sin_t_sq)
        phi = torch.atan2(s_yy, s_xx)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        sin_2phi = 2.0 * sin_phi * cos_phi
        cos_2phi = cos_phi ** 2 - sin_phi ** 2

        # Field after basis change
        e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * self.pupil.field[:, 0, :, :] \
                        + (cos_theta - 1.0) * sin_2phi * self.pupil.field[:, 1, :, :]
        e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * self.pupil.field[:, 1, :, :] \
                        + (cos_theta - 1.0) * sin_2phi * self.pupil.field[:, 0, :, :]
        e_inf_z = -2.0 * sin_theta * (cos_phi * self.pupil.field[:, 0, :, :] + sin_phi * self.pupil.field[:, 1, :, :])
        e_inf_x = torch.where(s_valid, e_inf_x, 0.0).unsqueeze(0) / 2
        e_inf_y = torch.where(s_valid, e_inf_y, 0.0).unsqueeze(0) / 2
        e_inf_z = torch.where(s_valid, e_inf_z, 0.0).unsqueeze(0) / 2
        self.e_inf_field = torch.cat((e_inf_x, e_inf_y, e_inf_z), dim=1)

        # Correction factors
        self.correction_factor = torch.ones(1, 1, n_pix_pupil, n_pix_pupil).to(torch.complex64).to(self.device)
        if self.sz_correction:
            self.correction_factor *= 1 / s_zz
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(s_zz)
        if self.envelope is not None:
            self.correction_factor *= torch.exp(- (1 - s_zz ** 2) / self.envelope ** 2)
        if self.gibson_lanni:
            clamp_value = np.minimum(self.n_s/self.n_i, self.n_g/self.n_i)
            sin_t = (self.s_max * torch.sqrt(s_xx**2 + s_yy**2)).clamp(max=clamp_value)
            path = self.compute_optical_path(sin_t)
            self.correction_factor *= torch.exp(1j * self.k * path)

        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus
                                       ).reshape(-1, 1, 1, 1).to(self.device)
        self.defocus_filters = torch.exp(1j * self.k * s_zz * defocus_range)

    def compute_focus_field(self):
        self.field = custom_ifft2(self.e_inf_field * self.correction_factor * self.defocus_filters,
                                  shape_out=(self.n_pix_psf, self.n_pix_psf),
                                  k_start=-self.zoom_factor * np.pi,
                                  k_end=self.zoom_factor * np.pi,
                                  norm='forward', fftshift_input=True, include_end=True) * (self.ds * self.s_max) ** 2
        return self.field / (2 * np.pi * np.sqrt(self.refractive_index))

    def _compute_psf_for_far_field(self, far_fields):  # to remove later?
        s_xx, s_yy = torch.meshgrid(self.s_x * self.s_max, self.s_x * self.s_max, indexing='ij')
        sin_t_sq = s_xx ** 2 + s_yy ** 2
        s_valid = sin_t_sq <= self.s_max ** 2
        sin_theta = torch.sqrt(sin_t_sq)
        cos_theta = torch.sqrt(1.0 - sin_t_sq)
        phi = torch.atan2(s_yy, s_xx)   # properly handles pole at sin_theta == 0.0
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        sin_2phi = 2.0 * sin_phi * cos_phi
        cos_2phi = cos_phi ** 2 - sin_phi ** 2

        far_fields_x = far_fields[:, 0, :, :]
        far_fields_y = far_fields[:, 1, :, :]
        e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * far_fields_x \
                        + (cos_theta - 1.0) * sin_2phi * far_fields_y
        e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * far_fields_y \
                        + (cos_theta - 1.0) * sin_2phi * far_fields_x
        e_inf_z = -2.0 * sin_theta * (cos_phi * far_fields_x + sin_phi * far_fields_y)

        e_inf_x = torch.where(s_valid, e_inf_x, 0.0).unsqueeze(0)
        e_inf_y = torch.where(s_valid, e_inf_y, 0.0).unsqueeze(0)
        e_inf_z = torch.where(s_valid, e_inf_z, 0.0).unsqueeze(0)

        PSF_field = custom_ifft2(torch.cat((e_inf_x, e_inf_y, e_inf_z), dim=1) * self.correction_factor * self.defocus_filters,
                                  shape_out=(self.n_pix_psf, self.n_pix_psf),
                                  k_start=-self.zoom_factor * np.pi,
                                  k_end=self.zoom_factor * np.pi,
                                  norm='forward', fftshift_input=True, include_end=True) \
                     * (self.ds * self.s_max) ** 2 * 1j
        PSF_field /= (2 * np.pi * np.sqrt(self.refractive_index))

        return PSF_field


class VectorialPolarPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, na=1.3, fov=1000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, t_i0=100e3,
                 quadrature_rule=simpsons_rule):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, na=na, fov=fov, refractive_index=refractive_index,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, t_i0=t_i0)
        # PSF coordinates
        x = torch.linspace(-self.fov / 2, self.fov / 2, self.n_pix_psf)
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        rr = torch.sqrt(xx ** 2 + yy ** 2)
        r_unique, rr_indices = torch.unique(rr, return_inverse=True)
        self.rs = r_unique.to(self.device)  # compute minimal number of points
        self.rr_indices = rr_indices.to(self.device)  # to invert

        # PSF varphi coordinate
        varphi = torch.atan2(yy, xx)
        sin_phi = torch.sin(varphi)
        cos_phi = torch.cos(varphi)
        sin_twophi = 2.0 * sin_phi * cos_phi
        cos_twophi = cos_phi ** 2 - sin_phi ** 2
        self.sin_phi = sin_phi.to(self.device)
        self.cos_phi = cos_phi.to(self.device)
        self.sin_twophi = sin_twophi.to(self.device)
        self.cos_twophi = cos_twophi.to(self.device)

        # Pupil coordinates
        theta_max = np.arcsin(self.na / self.refractive_index)
        num_thetas = self.n_pix_pupil
        thetas = torch.linspace(0, theta_max, num_thetas)
        dtheta = theta_max / (num_thetas - 1)
        self.thetas = thetas.to(self.device)
        self.dtheta = dtheta

        # Precompute additional factors
        self.k = 2.0 * np.pi / self.wavelength
        sin_t, cos_t = torch.sin(thetas), torch.cos(thetas)
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus)
        self.defocus_filters = torch.exp(1j * self.k * defocus_range[:,None] * cos_t[None,:]).to(self.device)   # [n_defocus, n_thetas]

        self.correction_factor = torch.ones(self.n_pix_pupil).to(torch.complex64).to(self.device)
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(cos_t)
        if self.envelope is not None:
            self.correction_factor *= torch.exp(-(sin_t / self.envelope) ** 2)
        if self.gibson_lanni:
            clamp_value = np.minimum(self.n_s/self.n_i, self.n_g/self.n_i)
            sin_t = sin_t.clamp(max=clamp_value)
            path = self.compute_optical_path(sin_t)
            self.correction_factor *= torch.exp(1j * self.k * path)
        self.quadrature_rule = quadrature_rule

    def compute_focus_field(self):
        # multiplicative scalar factor to be verified for the vectorial case
        self.field = self._compute_psf_for_far_field(self.pupil.field)
        return self.field

    def _compute_psf_for_far_field(self, far_fields):
        sin_t = torch.sin(self.thetas) # [n_thetas, ]
        cos_t = torch.cos(self.thetas) # [n_thetas, ]

        # bessel function evaluations are expensive and can be computed independently from defocus
        bessel_arg = self.k * self.rs[None,:] * sin_t[:,None]
        J0s = bessel_j0(bessel_arg)    # [n_theta, n_radii]
        J1s = bessel_j1(bessel_arg)    # [n_theta, n_radii]
        # bessel_j2() evaluations expressed in terms of j0(), j1()
        J2s = 2.0 * torch.where(bessel_arg > 1e-6,
                                J1s / bessel_arg,
                                0.5 - bessel_arg ** 2 / 16) - J0s

        # compute PSF field; handle defocus via batching with vmap()
        batched_compute_field_at_defocus = vmap(self._compute_psf_at_defocus, in_dims=(0, None, None, None, None, None, None))
        fields = batched_compute_field_at_defocus(self.defocus_filters, J0s, J1s, J2s, far_fields, sin_t, cos_t)
        return fields

    def _compute_psf_at_defocus(self, defocus_term, J0s, J1s, J2s, far_fields, sin_t, cos_t):
        field_x, field_y = far_fields[:, 0, :].squeeze(), far_fields[:, 1, :].squeeze()

        # compute E(r) for a list of unique radii values
        # shape(Ix0) == shape(Iy0) == ... == [n_radii,]
        # TODO: extend `quadrature_rule` to act on vector-valued functions?
        I_term = sin_t * (cos_t + 1.0) * defocus_term * self.correction_factor
        Ix0 = self.quadrature_rule(dx=self.dtheta,
            fs=J0s * (field_x * I_term)[:,None])
        Iy0 = self.quadrature_rule(dx=self.dtheta,
            fs=J0s * (field_y * I_term)[:,None])

        I_term = sin_t ** 2 * defocus_term * self.correction_factor
        Ix1 = self.quadrature_rule(dx=self.dtheta,
            fs=J1s * (field_x * I_term)[:,None])
        Iy1 = self.quadrature_rule(dx=self.dtheta,
            fs=J1s * (field_y * I_term)[:,None])

        I_term = sin_t * (cos_t - 1.0) * defocus_term * self.correction_factor
        Ix2 = self.quadrature_rule(dx=self.dtheta,
            fs=J2s * (field_x * I_term)[:,None])
        Iy2 = self.quadrature_rule(dx=self.dtheta,
            fs=J2s * (field_y * I_term)[:,None])

        # scatter the radial evaluations of E(r) onto the xy image grid
        # shape(Ix0) == shape(Iy0) == ... == [nx,ny]
        Ix0 = Ix0[self.rr_indices]
        Iy0 = Iy0[self.rr_indices]
        Ix1 = Ix1[self.rr_indices]
        Iy1 = Iy1[self.rr_indices]
        Ix2 = Ix2[self.rr_indices]
        Iy2 = Iy2[self.rr_indices]

        # updated expression with correct 1j factors
        PSF_field = torch.stack([                                   # [n_channels=3, size_x, size_y]
            Ix0 - Ix2 * self.cos_twophi - Iy2 * self.sin_twophi,
            Iy0 - Ix2 * self.sin_twophi + Iy2 * self.cos_twophi,
            -2j * (Ix1 * self.cos_phi + Iy1 * self.sin_phi)],
            dim=0)

        return PSF_field / 2 / np.sqrt(self.refractive_index)

