# Copyright Biomedical Imaging Group, EPFL 2024

"""
The propagator in the case of Cartesian coordinates.

"""

import math
from abc import ABC

import torch

from .propagator import Propagator
from utils.czt import custom_ifft2
from utils.zernike import create_zernike_aberrations


class CartesianPropagator(Propagator, ABC):
    def __init__(self, n_pix_pupil=128, n_pix_psf=128, device='cpu',
                 zernike_coefficients=None,
                 wavelength=632, na=1.3, fov=1000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 sz_correction=True, apod_factor=False, envelope=None,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, t_i0=100e3):
        super().__init__(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix_psf, device=device,
                         zernike_coefficients=zernike_coefficients,
                         wavelength=wavelength, na=na, fov=fov, refractive_index=refractive_index,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, t_i0=t_i0)
        self.sz_correction = sz_correction

        # Physical parameters
        self.k = 2 * torch.pi / self.wavelength
        self.s_max = torch.tensor(self.na / self.refractive_index)

         # Zoom factor to determine pixel size with custom FFT
        self.zoom_factor = 2 * self.s_max * self.fov / self.wavelength \
             / (self.n_pix_pupil - 1)

        # Coordinates in pupil space s_x, s_y, s_z
        n_pix_pupil = self.n_pix_pupil
        self.s_x = torch.linspace(-1, 1, n_pix_pupil).to(self.device)
        self.ds = self.s_x[1] - self.s_x[0]
        s_xx, s_yy = torch.meshgrid(self.s_x, self.s_x, indexing='ij')
        s_zz = torch.sqrt((1 - self.s_max ** 2 * (s_xx ** 2 + s_yy ** 2)
                          ).clamp(min=0.001)).reshape(1, 1, n_pix_pupil, n_pix_pupil)

        # Coordinates in object space
        total_fft_range = 1.0 / self.ds
        k_start = -self.zoom_factor * torch.pi
        k_end = self.zoom_factor * torch.pi
        self.x = torch.linspace(k_start, k_end, self.n_pix_pupil).to(self.device) / (2.0 * torch.pi) * total_fft_range

        # Correction factors
        self.correction_factor = torch.ones(1, 1, n_pix_pupil, n_pix_pupil).to(torch.complex64).to(self.device)
        if self.sz_correction:
            self.correction_factor *= 1 / s_zz
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(s_zz)
        if self.envelope is not None:
            self.correction_factor *= torch.exp(- (1 - s_zz ** 2) / self.envelope ** 2)
        if self.gibson_lanni:
            clamp_value = min(self.n_s/self.n_i, self.n_g/self.n_i)
            sin_t = (self.s_max * torch.sqrt(s_xx**2 + s_yy**2)).clamp(max=clamp_value)
            path = self.compute_optical_path(sin_t)
            self.correction_factor *= torch.exp(1j * self.k * path)

        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus
                                       ).reshape(-1, 1, 1, 1).to(self.device)
        self.defocus_filters = torch.exp(1j * self.k * s_zz * defocus_range)

    def _zernike_aberrations(self):
        """Compute Zernike aberrations that will be applied on the pupil."""
        aberrations = create_zernike_aberrations(self.zernike_coefficients, self.n_pix_pupil, mesh_type='cartesian')
        return aberrations.to(self.device)

    def compute_focus_field(self):
        """Compute the electric field at the focal plane."""
        input_field = self.get_input_field()
        field = custom_ifft2(input_field * self.correction_factor * self.defocus_filters,
                                  shape_out=(self.n_pix_psf, self.n_pix_psf),
                                  k_start=-self.zoom_factor * torch.pi,
                                  k_end=self.zoom_factor * torch.pi,
                                  norm='forward', fftshift_input=True, include_end=True) * (self.ds * self.s_max) ** 2
        return field / (2 * math.pi * math.sqrt(self.refractive_index))

