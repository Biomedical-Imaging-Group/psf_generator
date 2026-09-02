# Copyright Biomedical Imaging Group, EPFL 2025

"""
The propagator in the case of Spherical coordinates.

"""

import math
import warnings
from abc import ABC

import torch

from .propagator import Propagator
from ..utils import integrate
from ..utils.integrate import simpsons_rule


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

    - The pupil is sampled **uniformly in the polar angle** :math:`\theta`: the `n_pix_pupil` samples are
      :math:`\theta_i = i \, \theta_{\max} / (n_{\mathrm{pix}} - 1)` with
      :math:`\theta_{\max} = \arcsin(\mathrm{na} / n_i^0)`. Sample :math:`i` therefore sits at the normalized
      pupil radius

      .. math:: \rho_i = \frac{\sin\theta_i}{\sin\theta_{\max}},

      which is *not* :math:`i / (n_{\mathrm{pix}} - 1)`. The radii are stored in the attribute ``rho`` and the
      Zernike modes are evaluated there, so that a given set of coefficients describes the same wavefront as in
      the Cartesian propagators (which sample the pupil uniformly in :math:`\rho`). A `custom_field` is
      interpreted on the same :math:`\theta` grid.

    """

    _zernike_mesh_type = 'spherical'

    def __init__(self, n_pix_pupil=128, n_pix_psf=128, device='cpu',
                 zernike_coefficients=None,
                 custom_field=None,
                 wavelength=632, na=1.3, pix_size=10,
                 defocus_step=0, n_defocus=1,
                 apod_factor=False, envelope=None, cos_factor=False,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, n_i0=1.5, t_i0=100e3,
                 integrator=simpsons_rule):
        super().__init__(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix_psf, device=device,
                         zernike_coefficients=zernike_coefficients,
                         wavelength=wavelength, na=na, pix_size=pix_size,
                         defocus_step=defocus_step, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, n_i0=n_i0, t_i0=t_i0)
        # PSF coordinates (pixel-centred grid, see ``Propagator.x``)
        self.yy, self.xx = torch.meshgrid(self.x, self.x, indexing='ij')
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
        # Normalized radius of every pupil sample. The pupil is sampled uniformly in theta, so sample i sits at
        # rho_i = sin(theta_i) / sin(theta_max) and *not* at i / (n_pix_pupil - 1); the Zernike modes are
        # evaluated there (see ``_zernike_radius``). Computed in float64 and clamped to [0, 1] because
        # sin(arcsin(s_max)) / s_max may round above 1, which would zero the outermost sample of every mode.
        self.rho = (torch.sin(thetas.double()) / self.s_max.double()).clamp(0.0, 1.0)

        # Precompute additional factors
        self.cos_factor = cos_factor
        self.k = 2.0 * math.pi / self.wavelength
        sin_t, cos_t = torch.sin(thetas), torch.cos(thetas)

        defocus_range = self.z
        self.defocus_filters = torch.exp(1j * self.k * defocus_range[:, None] * cos_t[None, :] * self.refractive_index).to(self.device)   # [n_defocus, n_thetas]

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

        # custom field (1D array of length n_pix_pupil or None)
        if custom_field is not None:
            if not isinstance(custom_field, torch.Tensor):
                custom_field = torch.tensor(custom_field, dtype=torch.complex64)
            if custom_field.shape != (n_pix_pupil,):
                raise ValueError(f"custom_field must have shape ({n_pix_pupil},)")
            self.custom_field = custom_field.to(torch.complex64).to(self.device)
        else:
            self.custom_field = None

        # Numerical integration method
        self.integrator = integrator

        # Precompute Zernike aberrations
        self._compute_zernike_aberrations()

    def _zernike_radius(self) -> torch.Tensor:
        """Normalized radius :math:`\sin\theta_i / \sin\theta_{\max}` of every pupil sample (float64, CPU)."""
        return self.rho

    def update_custom_field(self, custom_field):
        """
        Update custom field without reinitializing propagator.

        Parameters
        ----------
        custom_field : torch.Tensor or None
            Custom field of shape (n_pix_pupil,).
        """
        if custom_field is None:
            self.custom_field = None
            return
        if not isinstance(custom_field, torch.Tensor):
            custom_field = torch.tensor(custom_field, dtype=torch.complex64)
        if custom_field.shape != (self.n_pix_pupil,):
            raise ValueError(f"custom_field must have shape ({self.n_pix_pupil},)")
        self.custom_field = custom_field.to(torch.complex64).to(self.device)

    def get_correction_factor(self):
        """
        Get the correction factor applied to the pupil (apod_factor, envelope, gibson_lanni, cos_factor).

        Returns
        -------
        torch.Tensor
            Correction factor of shape (n_pix_pupil,).
        """
        return self.correction_factor

    def _get_args(self) -> dict:
        args = super()._get_args()
        args['cos_factor'] = self.cos_factor
        args['integrator'] = self.integrator.__name__
        if self.custom_field is not None:
            warnings.warn('The custom_field tensor cannot be saved to JSON and is not stored.', stacklevel=3)
        return args

    @classmethod
    def _decode_args(cls, args: dict) -> dict:
        args = super()._decode_args(args)
        integrator = args.get('integrator')
        if isinstance(integrator, str):
            if integrator not in ('riemann_rule', 'trapezoid_rule', 'simpsons_rule'):
                raise ValueError(f'Unknown integrator {integrator!r}.')
            args['integrator'] = getattr(integrate, integrator)
        return args

    def get_pupil(self):
        """Get the pupil function with all corrections applied."""
        pupil = self.initialize_input_field()
        pupil = pupil * self._zernike_aberrations
        pupil = pupil * self.correction_factor
        if self.custom_field is not None:
            pupil = pupil * self.custom_field
        return pupil
