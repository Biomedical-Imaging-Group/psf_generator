import torch
import numpy as np
from abc import ABC, abstractmethod
from utils.czt_new import custom_ifft2

from utils.integrate import integrate_summation_rule
from torch.special import bessel_j0


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
                 sz_correction=True, apod_factor=True, envelope=None):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, NA=NA, fov=fov, 
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus)
        self.sz_correction = sz_correction
        self.apod_factor = apod_factor  # we may want to add more options, sometimes we need to divide by sqrt(s_z)
        self.envelope = envelope  # for Gaussian envelope
        
         # Zoom factor to determine pixel size with custom FFT
        self.zoom_factor = 2 * self.NA * self.fov / self.wavelength / self.n_pix_pupil

        # Compute coordinates s_x, s_y, s_z
        n_pix_pupil = self.pupil.n_pix_pupil
        x = torch.linspace(-1, 1, n_pix_pupil).to(self.device)
        s_x, s_y = torch.meshgrid(x, x, indexing='ij')
        s_z = torch.sqrt((1 - self.NA**2 * (s_x**2 + s_y**2)).clamp(min=0.001)).reshape(1, 1, n_pix_pupil, n_pix_pupil)

        # Precompute additional factors
        correction_factor = 1
        if self.sz_correction:
            correction_factor *= 1 / s_z
        if self.apod_factor:
            correction_factor *= torch.sqrt(s_z)
        if self.envelope is not None:
            correction_factor *= torch.exp(- (1-s_z**2) / self.envelope**2)
        self.correction_factor = correction_factor.to(self.device)
        self.k = 2 * np.pi / self.wavelength
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus).reshape(-1, 1, 1, 1) 
        self.defocus_filters = torch.exp(1j * self.k * s_z * defocus_range).to(self.device)

    def compute_focus_field(self):
        self.field = custom_ifft2(self.pupil.field * self.correction_factor * self.defocus_filters, 
                                  shape_out=(self.n_pix_psf, self.n_pix_psf), 
                                  k_start=-self.zoom_factor*np.pi, 
                                  k_end=self.zoom_factor*np.pi, 
                                  norm='ortho', fftshift_input=True)
        return self.field


class ScalarPolarPropagator(Propagator):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=0.9, fov=1000, 
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=True, envelope=None):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, NA=NA, fov=fov, 
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus)
        self.apod_factor = apod_factor
        self.envelope = envelope
        
        # PSF coordinates
        x = torch.linspace(-self.fov/2, self.fov/2, self.n_pix_psf)
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        self.r = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(self.device)

        # Pupil coordinates
        theta_max = np.arcsin(self.NA)
        theta = torch.linspace(0, theta_max, self.n_pix_pupil).to(self.device)

        # Precompute additional factors
        self.k = 2 * np.pi / self.wavelength
        self.sin_t = torch.reshape(torch.sin(theta), (1, 1, 1, 1, -1)).to(self.device)
        cos_t = torch.reshape(torch.cos(theta), (1, 1, 1, 1, -1))
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus).reshape(-1, 1, 1, 1, 1) 
        self.defocus_filters = torch.exp(1j * self.k * defocus_range * cos_t).to(self.device)
        correction_factor = 1
        if self.apod_factor:
            correction_factor *= torch.sqrt(cos_t)
        if self.envelope is not None:
            correction_factor *= torch.exp(- self.sin_t**2 / self.envelope**2)
        self.correction_factor = correction_factor.to(self.device)

    def compute_focus_field(self):
        self.field = torch.sum(
            self.pupil.field.unsqueeze(-2).unsqueeze(-2) *
            bessel_j0(self.k * self.r * self.sin_t) *
            self.sin_t * self.defocus_filters * self.correction_factor
            , dim=-1)
        return self.field


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

