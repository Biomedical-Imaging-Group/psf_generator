import torch
import numpy as np
from abc import ABC, abstractmethod
from utils.czt import custom_ifft2

from utils.integrate import integrate_summation_rule
from torch.special import bessel_j0, bessel_j1
from scipy.special import itj0y0

from integrators import trapezoid_rule, simpsons_rule, richard2_rule
# # re-enable if gradients wrt Bessel term are required
# from bessel_ad import BesselJ0
# bessel_j0_ad = BesselJ0.apply


class Propagator(ABC):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=0.9, fov=2000, 
                 defocus_min=0, defocus_max=0, n_defocus=1):
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
        self.defocus_min = defocus_min
        self.defocus_max = defocus_max
        self.n_defocus = n_defocus

        self.field = None

    @abstractmethod
    def compute_focus_field(self):
        raise NotImplementedError

    
class ScalarCartesianPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=0.9, fov=1000, 
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 sz_correction=True, apod_factor=False, envelope=None):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, NA=NA, fov=fov, 
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus)
        self.sz_correction = sz_correction
        self.apod_factor = apod_factor  # we may want to add more options, sometimes we need to divide by sqrt(s_z)
        self.envelope = envelope        # for Gaussian envelope
        
         # Zoom factor to determine pixel size with custom FFT
        self.zoom_factor = 2 * self.NA * self.fov / self.wavelength / self.n_pix_pupil

        # Compute coordinates s_x, s_y, s_z
        n_pix_pupil = self.pupil.n_pix_pupil
        x = torch.linspace(-1, 1, n_pix_pupil).to(self.device)
        s_x, s_y = torch.meshgrid(x, x, indexing='ij')
        s_z = torch.sqrt((1 - self.NA**2 * (s_x**2 + s_y**2)).clamp(min=0.001)).reshape(1, 1, n_pix_pupil, n_pix_pupil)

        # Precompute additional factors
        self.correction_factor = torch.ones(1, 1, n_pix_pupil, n_pix_pupil).to(self.device)
        if self.sz_correction:
            self.correction_factor *= 1 / s_z
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(s_z)
        if self.envelope is not None:
            self.correction_factor *= torch.exp(- (1-s_z**2) / self.envelope**2)
        self.k = 2 * np.pi / self.wavelength
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus
                                       ).reshape(-1, 1, 1, 1).to(self.device)
        self.defocus_filters = torch.exp(1j * self.k * s_z * defocus_range)

    def compute_focus_field(self):
        self.field = custom_ifft2(self.pupil.field * self.correction_factor * self.defocus_filters, 
                                  shape_out=(self.n_pix_psf, self.n_pix_psf), 
                                  k_start=-self.zoom_factor*np.pi, 
                                  k_end=self.zoom_factor*np.pi, 
                                  norm='backward', fftshift_input=True, include_end=True) * \
                                    (2 * self.NA / self.n_pix_pupil)**2
        return self.field / (2 * np.pi)


class ScalarPolarPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=0.9, fov=1000, 
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None, quadrature_rule=simpsons_rule):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, NA=NA, fov=fov, 
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus)
        self.apod_factor = apod_factor
        self.envelope = envelope
        
        # PSF coordinates
        x = torch.linspace(-self.fov/2, self.fov/2, self.n_pix_psf)
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        rr = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(0).unsqueeze(0).to(self.device)
        r_unique, rr_indices = torch.unique(rr, return_inverse=True)
        self.rs = r_unique.to(self.device)
        self.rr_indices = rr_indices.to(self.device)

        # Pupil coordinates
        # TODO: number of pixels in pupil (== gridsize of integration domain) should be driven
        # by the integration method and its required accuracy, not set a-priori
        # TODO: ideally, pupil should be described by a *continuous function* that allows us to query its
        # value at any value of `theta`
        theta_max = np.arcsin(self.NA)
        num_thetas = self.n_pix_pupil
        thetas = torch.linspace(0, theta_max, num_thetas).to(self.device)
        dtheta = theta_max / (num_thetas - 1)
        self.thetas = thetas.to(self.device)
        self.dtheta = dtheta

        # Precompute additional factors
        self.k = 2.0 * np.pi / self.wavelength
        sin_t, cos_t = torch.sin(thetas), torch.cos(thetas)
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus)
        self.defocus_filters = torch.exp(1j * self.k * defocus_range[:,None] * cos_t[None,:]).to(self.device)   # [n_defocus, n_thetas]

        correction_factor = torch.ones(num_thetas)
        if self.apod_factor:
            correction_factor *= torch.sqrt(cos_t)
        if self.envelope is not None:
            correction_factor *= torch.exp(-(sin_t / self.envelope) ** 2)
        self.correction_factor = correction_factor.to(self.device)
        self.quadrature_rule = quadrature_rule

    def compute_focus_field(self):
        # argument shapes:
        # self.thetas,            [n_thetas, ]
        # self.dtheta,            float
        # self.rs,                [n_radii, ]
        # self.correction_factor  [n_thetas, ]

        sin_t = torch.sin(self.thetas) # [n_thetas, ]
        far_fields = self.pupil.field.squeeze()   # [n_defocus=1, channels=1, n_thetas] ==> [n_thetas, ]
        
        # bessel function evaluations are expensive and can be computed independently from defocus
        J_evals = bessel_j0(self.k * self.rs[None,:] * sin_t[:,None])    # [n_theta, n_radii]

        # compute PSF field; handle defocus with an outer loop (not parallelized)
        fields = torch.zeros_like(self.rr_indices, dtype=torch.complex64)   # [n_defocus, channels, size_x, size_y]
        # TODO: replace loop with vmap()?
        for i in range(len(self.defocus_filters)):
            defocus_term = self.defocus_filters[i]  #[n_thetas, ]
            # compute E(r) for a list of unique radii values
            integrand = J_evals * (far_fields * defocus_term * self.correction_factor * sin_t)[:,None]  # [n_theta, n_radii]
            field = self.quadrature_rule(integrand, self.dtheta)
            # field = riemann_rule(integrand, self.dtheta)
            # scatter the radial evaluations of E(r) onto the xy image grid
            fields[i,:] = field[self.rr_indices]

        # shape: [`n_defocus` x `channels` x `size_x` x `size_y`]
        return fields


    def test_compute_field_tc1(self):
        sin_t = torch.sin(self.thetas)
        J_evals = bessel_j0(self.k * self.rs[None,:] * sin_t[:,None])
        # MODIFICATION FOR TEST CASE: put r**2 factor in integrand
        J_evals *= self.rs[None,:] ** 2
        # MODIFICATION FOR TEST CASE: far field corresponding to analytic solution
        far_fields = self.k ** 2 * torch.cos(self.thetas)

        fields = torch.zeros_like(self.rr_indices, dtype=torch.complex64)
        fields_sol = torch.zeros_like(fields, dtype=torch.complex64)
        # for i in range(len(self.defocus_filters)):
        #     defocus_term = self.defocus_filters[i]
        # MODIFICATION FOR TEST CASE: drop defocus and correction (apod + envelope) terms
        integrand = J_evals * (far_fields * sin_t)[:,None]
        field = self.quadrature_rule(integrand, self.dtheta)
        fields[0] = field[self.rr_indices]

        sol_arg = self.k * self.rs * self.NA    # NA == torch.sin(self.theta_max)
        field_sol = sol_arg * bessel_j1(sol_arg)
        fields_sol[0] = field_sol[self.rr_indices]

        return fields, fields_sol

    def test_compute_field_tc2(self):
        sin_t = torch.sin(self.thetas)
        J_evals = bessel_j0(self.k * self.rs[None,:] * sin_t[:,None])
        # MODIFICATION FOR TEST CASE: put `r` factor in integrand
        J_evals *= self.rs[None,:]
        # MODIFICATION FOR TEST CASE: far field corresponding to analytic solution
        far_fields = self.k * torch.cos(self.thetas)

        fields = torch.zeros_like(self.rr_indices, dtype=torch.complex64)
        fields_sol = torch.zeros_like(fields, dtype=torch.complex64)
        # MODIFICATION FOR TEST CASE: drop defocus and correction (apod + envelope) terms
        # MODIFICATION FOR TEST CASE: far field cancels the `sin(t)` term in the integrand
        integrand = J_evals * (far_fields)[:,None]
        field = self.quadrature_rule(integrand, self.dtheta)
        fields[0] = field[self.rr_indices]

        sol_arg = self.k * self.rs * self.NA    # NA == torch.sin(self.theta_max)
        field_sol = torch.tensor(itj0y0(sol_arg.numpy())[0], dtype=torch.complex64).to(self.device)
        fields_sol[0] = field_sol[self.rr_indices]

        return fields, fields_sol






class Vectorial(Propagator):
    """Richards-Wolf model (vectorial). """

    def __init__(self, pupil, params):
        super().__init__(pupil, params)

        self.pupil = pupil
        self.params = params
        self.field = None

        if pupil is None:
            self.pupil = VectorialPupil(params)

    def compute_focus_field(self):
        """Compute the vectorial field at focus.
        """

        pupil = self.pupil.return_pupil()
        size = self.params.get('n_pix_pupil')
        x = torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        y = torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        z =  torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        xx, yy, zz  = torch.meshgrid(x, y, z,  indexing='ij')
        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))

        i0_x = integrate_summation_rule(lambda theta: self.integrand00(theta, xx, yy, zz, pupil[0]), 0, theta_max, size)
        i2_x = integrate_summation_rule(lambda theta: self.integrand02(theta, xx, yy, zz, pupil[0]), 0, theta_max, size)
        i1_x = integrate_summation_rule(lambda theta: self.integrand01(theta, xx, yy, zz, pupil[0]), 0, theta_max, size)

        i0_y = integrate_summation_rule(lambda theta: self.integrand00(theta, xx, yy, zz, pupil[1]), 0, theta_max, size)
        i2_y = integrate_summation_rule(lambda theta: self.integrand02(theta, xx, yy, zz, pupil[1]), 0, theta_max, size)
        i1_y = integrate_summation_rule(lambda theta: self.integrand01(theta, xx, yy, zz, pupil[1]), 0, theta_max, size)

        varphi = torch.atan2(yy, xx)
        field_x = i0_x + i2_x * torch.cos(2 * varphi) + i2_y * torch.sin(2 * varphi)
        field_y = i2_x * torch.sin(2 * varphi) + i0_y - i2_y * torch.cos(2 * varphi)
        field_z = -2 * 1j * i1_x * torch.cos(varphi) - 2 * 1j * i1_y * torch.sin(varphi)

        self.field = torch.stack((field_x, field_y, field_z), dim=0).movedim(-1,0)

        return torch.abs(self.field) ** 2

    def integrand00(self, theta, xx, yy, zz, pupil):
        pupil = self.pupil.return_pupil()
        size = self.params.get('n_pix_pupil')
        x = torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        y = torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        z =  torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        xx, yy, zz  = torch.meshgrid(x, y, z,  indexing='ij')
        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))

        i0_x = integrate_summation_rule(lambda theta: self.integrand00(theta, xx, yy, zz, pupil[0]), 0, theta_max, size)
        i2_x = integrate_summation_rule(lambda theta: self.integrand02(theta, xx, yy, zz, pupil[0]), 0, theta_max, size)
        i1_x = integrate_summation_rule(lambda theta: self.integrand01(theta, xx, yy, zz, pupil[0]), 0, theta_max, size)

        i0_y = integrate_summation_rule(lambda theta: self.integrand00(theta, xx, yy, zz, pupil[1]), 0, theta_max, size)
        i2_y = integrate_summation_rule(lambda theta: self.integrand02(theta, xx, yy, zz, pupil[1]), 0, theta_max, size)
        i1_y = integrate_summation_rule(lambda theta: self.integrand01(theta, xx, yy, zz, pupil[1]), 0, theta_max, size)

        varphi = torch.atan2(yy, xx)
        field_x = i0_x + i2_x * torch.cos(2 * varphi) + i2_y * torch.sin(2 * varphi)
        field_y = i2_x * torch.sin(2 * varphi) + i0_y - i2_y * torch.cos(2 * varphi)
        field_z = -2 * 1j * i1_x * torch.cos(varphi) - 2 * 1j * i1_y * torch.sin(varphi)

        self.field = torch.stack((field_x, field_y, field_z), dim=0).movedim(-1,0)

        return torch.abs(self.field) ** 2

    def integrand00(self, theta, xx, yy, zz, pupil):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j0 = sp.bessel_j0(r)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j0 = sp.bessel_j0(r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t * (1 + cos_t)

        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]


        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]

        return torch.multiply(i, j0)

    def integrand02(self, theta, xx, yy, zz, pupil):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        eps = 1e-10
        j2 = 2 * sp.bessel_j1(r) / (r+eps) - sp.bessel_j0(r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t * (1 - cos_t)

        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]

        return torch.multiply(i, j2)

    def integrand01(self, theta, xx, yy, zz, pupil):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j1 = sp.bessel_j1(r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t ** 2

        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]

        return torch.multiply(i, j1)

    def integrand02(self, theta, xx, yy, zz, pupil):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        eps = 1e-10
        j2 = 2 * sp.bessel_j1(r) / (r+eps) - sp.bessel_j0(r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t * (1 - cos_t)

        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]

        return torch.multiply(i, j2)

    def integrand01(self, theta, xx, yy, zz, pupil):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j1 = sp.bessel_j1(r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t ** 2

        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]

        return torch.multiply(i, j1)

