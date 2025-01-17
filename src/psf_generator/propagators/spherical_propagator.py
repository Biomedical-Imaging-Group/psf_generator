# Copyright Biomedical Imaging Group, EPFL 2025

"""
The propagator in the case of Spherical coordinates.

"""

import math
from abc import ABC

import torch

from .propagator import Propagator
from ..utils.integrate import simpsons_rule
from ..utils.zernike import create_zernike_aberrations


class SphericalPropagator(Propagator, ABC):
    r"""
    Intermediate class for propagators with spherical parameterization.

    Notes
    -----
    - Apart from parameters inherited from the base class, there is one additional
      `cos_factor`. This cosine factor is only here to make the spherical propagator
      equivalent to the Cartesian propagator when sz_correction is set to False. 
      This is useful to compute analytic low NA PSFs such as the Airy disk. 


    - The spherical propagator makes the assumption that the input field (pupil) is axisymmetric (rotational-invariant).
      In other words, the input field is function of theta only and not dependent on the angle phi:

      .. math:: \mathbf{e}_{\infty}(\theta, \phi) = \mathbf{e}_{\infty}(\theta).

    """

    def __init__(self, n_pix_pupil=128, n_pix_psf=128, device='cpu',
                 zernike_coefficients=None,
                 wavelength=632, na=1.3, fov=1000,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None, cos_factor=False,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, n_i0=1.5, t_i0=100e3,
                 integrator=simpsons_rule):
        super().__init__(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix_psf, device=device,
                         zernike_coefficients=zernike_coefficients,
                         wavelength=wavelength, na=na, fov=fov,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, n_i0=n_i0, t_i0=t_i0)
        # PSF coordinates
        x = torch.linspace(-self.fov / 2, self.fov / 2, self.n_pix_psf)
        self.xx, self.yy = torch.meshgrid(x, x, indexing='ij')
        rr = torch.sqrt(self.xx ** 2 + self.yy ** 2)
        r_unique, rr_indices = torch.unique(rr, return_inverse=True)
        self.rs = r_unique.to(self.device)  # compute minimal number of points
        self.rr_indices = rr_indices.to(self.device)  # to invert

        # Pupil coordinates
        self.s_max = torch.tensor(self.na / self.n_i0)
        theta_max = torch.arcsin(self.s_max)
        num_thetas = self.n_pix_pupil
        thetas = torch.linspace(0, theta_max, num_thetas)
        self.thetas = thetas.to(self.device)
        dtheta = theta_max / (num_thetas - 1)
        self.dtheta = dtheta

        # Precompute additional factors
        self.cos_factor = cos_factor
        self.k = 2.0 * math.pi / self.wavelength
        sin_t, cos_t = torch.sin(thetas), torch.cos(thetas)

        defocus_range = torch.linspace(self.defocus_min, self.defocus_max, self.n_defocus)
        self.defocus_filters = torch.exp(1j * self.k * defocus_range[:,None] * cos_t[None,:] * self.refractive_index).to(self.device)   # [n_defocus, n_thetas]

        self.correction_factor = torch.ones(self.n_pix_pupil).to(torch.complex64).to(self.device)
        if self.apod_factor:
            self.correction_factor *= torch.sqrt(cos_t)
        if self.envelope is not None:
            self.correction_factor *= torch.exp(-(sin_t / self.envelope) ** 2)
        if self.gibson_lanni:
            clamp_value = min(self.n_s/self.n_i, self.n_g/self.n_i)
            sin_t = sin_t.clamp(max=clamp_value)
            path = self.compute_optical_path(sin_t)
            self.correction_factor *= torch.exp(1j * self.k * path)
        if self.cos_factor:
            self.correction_factor *= cos_t

        # Numerical integration method
        self.integrator = integrator

    def _aberrations(self) -> torch.Tensor:
        """Compute Zernike aberrations that will be applied on the pupil."""
        zernike_aberrations = create_zernike_aberrations(self.zernike_coefficients, self.n_pix_pupil, mesh_type='spherical').to(self.device)
        aberrations = zernike_aberrations * self.correction_factor
        return aberrations
