import torch
import numpy as np
from abc import ABC, abstractmethod
from utils.czt import custom_ifft2

from utils.integrate import integrate_summation_rule
from torch.special import bessel_j0, bessel_j1


class Propagator(ABC):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, NA=1.3, fov=2000, refractive_index=1.5, 
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None, 
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3, 
                 n_i=1.5, n_i0=1.5, t_i0=100e3):
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
        self.n_i0 = n_i0
        self.t_i0 = t_i0
        self.t_i = n_i * (t_g0/n_g0 + t_i0/n_i0 - t_g/n_g - z_p/n_s)

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
                 n_i=1.5, n_i0=1.5, t_i0=100e3):
        super().__init__(pupil=pupil, n_pix_psf=n_pix_psf, device=device,
                         wavelength=wavelength, NA=NA, fov=fov, refractive_index=refractive_index, 
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope, 
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=1.3,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, n_i0=n_i0, t_i0=t_i0)
        self.sz_correction = sz_correction
        
         # Zoom factor to determine pixel size with custom FFT
        self.zoom_factor = 2 * self.NA * self.fov / self.wavelength \
             / self.refractive_index / self.n_pix_pupil

        # Compute coordinates s_x, s_y, s_z
        n_pix_pupil = self.pupil.n_pix_pupil
        x = torch.linspace(-1, 1, n_pix_pupil).to(self.device)
        s_x, s_y = torch.meshgrid(x, x, indexing='ij')
        s_z = torch.sqrt((1 - (self.NA/self.refractive_index)**2 * (s_x**2 + s_y**2)
                          ).clamp(min=0.001)).reshape(1, 1, n_pix_pupil, n_pix_pupil)

        # Precompute additional factors
        self.k = 2 * np.pi / self.wavelength
        self.correction_factor = torch.ones(1, 1, n_pix_pupil, n_pix_pupil
                                            ).to(torch.complex64).to(self.device)
        if self.sz_correction:
            self.correction_factor *= 1 / s_z
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(s_z)
        if self.envelope is not None:
            self.correction_factor *= torch.exp(- (1-s_z**2) / self.envelope**2)
        if self.gibson_lanni:
            # computed following Eq. (3.45) of François Aguet's thesis
            sin_t = (self.NA / self.refractive_index * torch.sqrt(s_x**2 + s_y**2)).clamp(max=1)
            optical_path = self.z_p * torch.sqrt(self.n_s**2 - self.n_i**2 * sin_t**2) \
                            + self.t_i * torch.sqrt(self.n_i**2 - self.n_i**2 * sin_t**2) \
                            - self.t_i0 * torch.sqrt(self.n_i0**2 - self.n_i**2 * sin_t**2) \
                            + self.t_g * torch.sqrt(self.n_g**2 - self.n_i**2 * sin_t**2) \
                            - self.t_g0 * torch.sqrt(self.n_g0**2 - self.n_i**2 * sin_t**2)
            self.correction_factor *= torch.exp(1j * self.k * optical_path)
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus
                                       ).reshape(-1, 1, 1, 1).to(self.device)
        self.defocus_filters = torch.exp(1j * self.k * s_z * defocus_range)

    def compute_focus_field(self):
        self.field = custom_ifft2(self.pupil.field * self.correction_factor * self.defocus_filters, 
                                  shape_out=(self.n_pix_psf, self.n_pix_psf), 
                                  k_start=-self.zoom_factor*np.pi, 
                                  k_end=self.zoom_factor*np.pi, 
                                  norm='backward', fftshift_input=True, include_end=True) * \
                                    (2 * self.NA / self.refractive_index / self.n_pix_pupil)**2
        return self.field / (2 * np.pi) / np.sqrt(self.refractive_index)


class ScalarPolarPropagator(Propagator):
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
                         n_i=n_i, n_i0=n_i0, t_i0=t_i0)
        
        # PSF coordinates
        x = torch.linspace(-self.fov/2, self.fov/2, self.n_pix_psf)
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        self.r = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(self.device)

        # Pupil coordinates
        self.theta_max = np.arcsin(self.NA / self.refractive_index)
        theta = torch.linspace(0, self.theta_max, self.n_pix_pupil).to(self.device)

        # Precompute additional factors
        self.k = 2 * np.pi / self.wavelength
        self.sin_t = torch.reshape(torch.sin(theta), (1, 1, 1, 1, -1))
        cos_t = torch.reshape(torch.cos(theta), (1, 1, 1, 1, -1))
        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus
                                       ).reshape(-1, 1, 1, 1, 1).to(self.device)
        self.defocus_filters = torch.exp(1j * self.k * defocus_range * cos_t)
        correction_factor = torch.ones(1, 1, 1, 1, self.n_pix_pupil).to(torch.complex64)
        if self.apod_factor:
            correction_factor *= torch.sqrt(cos_t)
        if self.envelope is not None:
            correction_factor *= torch.exp(- self.sin_t**2 / self.envelope**2)
        if self.gibson_lanni:
            # computed following Eq. (3.45) of François Aguet's thesis
            optical_path = self.z_p * torch.sqrt(self.n_s**2 - self.n_i**2 * self.sin_t**2) \
                            + self.t_i * torch.sqrt(self.n_i**2 - self.n_i**2 * self.sin_t**2) \
                            - self.t_i0 * torch.sqrt(self.n_i0**2 - self.n_i**2 * self.sin_t**2) \
                            + self.t_g * torch.sqrt(self.n_g**2 - self.n_i**2 * self.sin_t**2) \
                            - self.t_g0 * torch.sqrt(self.n_g0**2 - self.n_i**2 * self.sin_t**2)
            correction_factor *= torch.exp(1j * self.k * optical_path)
        self.correction_factor = correction_factor.to(self.device)

    def compute_focus_field(self):
        self.field = torch.sum(
            self.pupil.field.unsqueeze(-2).unsqueeze(-2) *
            bessel_j0(self.k * self.r * self.sin_t) *
            self.sin_t * self.defocus_filters * self.correction_factor
            , dim=-1) * self.theta_max / self.n_pix_pupil / np.sqrt(self.refractive_index)
        return self.field


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
                         n_i=n_i, n_i0=n_i0, t_i0=t_i0)

        # PSF coordinates
        x = torch.linspace(-self.fov / 2, self.fov / 2, self.n_pix_psf)
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        self.r = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(self.device)
        self.varphi = torch.atan2(yy, xx)

        # Pupil coordinates
        self.theta_max = np.arcsin(self.NA / self.refractive_index)
        theta = torch.linspace(0, self.theta_max, self.n_pix_pupil).to(self.device)

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
        # to be verified for the vectorial case
        if self.envelope is not None:
            correction_factor *= torch.exp(- self.sin_t ** 2 / self.envelope ** 2)
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
