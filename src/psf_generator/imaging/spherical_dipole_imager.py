# Copyright Biomedical Imaging Group, EPFL 2025

"""
The dipole imager in spherical coordinates (Bessel integrals).
"""
import math

import torch
from torch.special import bessel_j0, bessel_j1

from .dipole_imager import DipoleImager
from ..utils import integrate
from ..utils.integrate import simpsons_rule


def _complex_matmul_real(weights: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Product of a complex matrix with a real one, written with real products (works on every device)."""
    return torch.complex(weights.real @ kernel, weights.imag @ kernel)


class SphericalDipoleImager(DipoleImager):
    r"""
    Dipole imager in spherical coordinates: the azimuthal integral is done analytically with Bessel functions.

    The transverse components of the image field of a dipole :math:`\mathbf{p}` are

    .. math::

        \hat{E}_x = -\mathrm{i} k_i \left[ \frac{p_x}{2} (I_0 - I_2 \cos 2\varphi) - \frac{p_y}{2} I_2 \sin 2\varphi
        + \mathrm{i} p_z I_1 \cos\varphi \right], \qquad
        \hat{E}_y = -\mathrm{i} k_i \left[ -\frac{p_x}{2} I_2 \sin 2\varphi + \frac{p_y}{2} (I_0 + I_2 \cos 2\varphi)
        + \mathrm{i} p_z I_1 \sin\varphi \right],

    with :math:`(\rho, \varphi)` the polar coordinates of the pixel relative to the lateral position of the dipole
    and

    .. math::

        I_m(\rho) = \int_0^{\theta_{\max}} A_m(\theta) \, \mathrm{e}^{\mathrm{i} k \Lambda(\theta)} \,
        \mathrm{e}^{\mathrm{i} W(\theta)} \sqrt{\cos\theta} \sin\theta \, J_m(k_i \rho \sin\theta) \,
        \mathrm{d}\theta, \qquad m = 0, 1, 2,

    where :math:`A_0 = t_p \cos\theta_s + t_s`, :math:`A_1 = t_p \sin\theta_s`, :math:`A_2 = t_p \cos\theta_s - t_s`
    and :math:`W` is the (axisymmetric) Zernike aberration. In a homogeneous medium the integrals reduce to those
    of the Richards-Wolf focus field with the apodization factor, so the image of a dipole equals the focus field
    of the ``VectorialSphericalPropagator`` with ``apod_factor=True``.

    The pupil is sampled uniformly in :math:`\theta` (attribute ``thetas``) and the integral is evaluated with
    `integrator` (Simpson's rule by default, which needs an odd `n_pix_pupil`). The Bessel functions are only
    evaluated at the distinct radii of the (shifted) image grid, and all the axial positions sharing a lateral
    position are batched in a single matrix product.

    Parameters
    ----------
    custom_field : torch.Tensor or array-like, optional
        Extra complex factor applied to the pupil, of shape `(n_pix_pupil,)` on the :math:`\theta` grid.
    integrator : callable, optional
        Quadrature rule from :mod:`psf_generator.utils.integrate`. Default is `simpsons_rule`.

    See :class:`DipoleImager` for the other parameters.

    """

    _zernike_mesh_type = 'spherical'

    def __init__(self, n_pix_pupil=128, n_pix_psf=128, device='cpu',
                 zernike_coefficients=None,
                 custom_field=None,
                 wavelength=632, na=1.3, pix_size=20, z_focus=0.0,
                 n_s=1.33, n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, n_i0=1.5, t_i0=100e3,
                 fresnel='reciprocal',
                 integrator=simpsons_rule):
        super().__init__(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix_psf, device=device,
                         zernike_coefficients=zernike_coefficients,
                         wavelength=wavelength, na=na, pix_size=pix_size, z_focus=z_focus,
                         n_s=n_s, n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, n_i0=n_i0, t_i0=t_i0, fresnel=fresnel)
        # Image grid (CPU): dim 0 is y, dim 1 is x.
        self.yy, self.xx = torch.meshgrid(self.x, self.x, indexing='ij')

        # Pupil: uniform in the polar angle theta.
        theta_max = math.asin(self.s_max)
        thetas = torch.linspace(0.0, theta_max, n_pix_pupil, dtype=torch.float64)
        self.dtheta = theta_max / (n_pix_pupil - 1)
        self.thetas = thetas.to(torch.float32).to(self.device)
        # Normalized pupil radius of every sample, for the Zernike modes (see ``SphericalPropagator.rho``).
        self.rho = (torch.sin(thetas) / self.s_max).clamp(0.0, 1.0)
        sin_t, cos_t = torch.sin(thetas), torch.cos(thetas)
        self._sin_t = sin_t.to(torch.float32).to(self.device)

        # Quadrature weights folded into the angular kernels (every rule is linear).
        self.integrator = integrator
        weights = integrator(torch.eye(n_pix_pupil, dtype=torch.float64), self.dtheta)

        # z-independent part of the integrands: A_m(theta) exp(ik Lambda_0) sqrt(cos) sin * weights.
        factors = self._layer_factors(sin_t)
        base = torch.exp(1j * self.k * factors['path0']) * torch.sqrt(cos_t) * sin_t * weights
        a_0, a_1, a_2 = self._pattern_factors(factors)
        self._kernels = (torch.stack([a_0, a_1, a_2]) * base).to(torch.complex64).to(self.device)
        self._path_z = factors['path_z'].to(torch.complex64).to(self.device)

        self.update_custom_field(custom_field)
        self._compute_zernike_aberrations()

    @classmethod
    def get_name(cls) -> str:
        return 'spherical'

    def _zernike_radius(self) -> torch.Tensor:
        return self.rho

    def update_custom_field(self, custom_field) -> None:
        """
        Update the custom pupil factor without reinitializing the imager.

        Parameters
        ----------
        custom_field : torch.Tensor or None
            Complex factor of shape (n_pix_pupil,) on the :math:`\theta` grid, or None.
        """
        if custom_field is None:
            self.custom_field = None
            return
        if not isinstance(custom_field, torch.Tensor):
            custom_field = torch.tensor(custom_field, dtype=torch.complex64)
        if custom_field.shape != (self.n_pix_pupil,):
            raise ValueError(f'custom_field must have shape ({self.n_pix_pupil},)')
        self.custom_field = custom_field.to(torch.complex64).to(self.device)

    def get_pupil_factor(self) -> torch.Tensor:
        """Zernike and custom factors of the pupil, of shape `(n_pix_pupil,)` on the :math:`\theta` grid."""
        factor = self._zernike_aberrations
        if self.custom_field is not None:
            factor = factor * self.custom_field
        return factor

    def _get_args(self) -> dict:
        args = super()._get_args()
        args['integrator'] = self.integrator.__name__
        self._warn_custom_field_not_saved()
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

    def _compute_image_xy(self, dipole: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        device = self.device
        n_pix = self.n_pix_psf
        n_positions = positions.shape[0]
        k_i = self.k * self.n_i
        p_x, p_y, p_z = dipole.to(torch.complex64).to(device)

        # Angular weights of every position: kernels * pupil factor * exp(ik z_p sqrt(n_s^2 - n_i^2 sin^2)).
        z_p = positions[:, 2].to(torch.float32).to(device).to(torch.complex64)
        axial_filters = torch.exp(1j * self.k * z_p[:, None] * self._path_z[None, :])   # [n_positions, n_thetas]
        kernels = self._kernels * self.get_pupil_factor()[None, :]                       # [3, n_thetas]
        weights = kernels[None, :, :] * axial_filters[:, None, :]                        # [n_positions, 3, n_thetas]

        image = torch.empty((n_positions, 2, n_pix, n_pix), dtype=torch.complex64, device=device)
        laterals, groups = torch.unique(positions[:, :2], dim=0, return_inverse=True)
        for group, (x_p, y_p) in enumerate(laterals.tolist()):
            indices = torch.nonzero(groups == group, as_tuple=True)[0]
            # Polar coordinates of the pixels relative to the dipole; Bessel functions on the distinct radii only.
            dx, dy = self.xx - x_p, self.yy - y_p
            r_unique, r_indices = torch.unique(torch.sqrt(dx ** 2 + dy ** 2), return_inverse=True)
            r_unique, r_indices = r_unique.to(device), r_indices.to(device)
            bessel_arg = k_i * r_unique[None, :] * self._sin_t[:, None]                  # [n_thetas, n_radii]
            J0 = bessel_j0(bessel_arg)
            J1 = bessel_j1(bessel_arg)
            J2 = 2.0 * torch.where(bessel_arg > 1e-6, J1 / bessel_arg, 0.5 - bessel_arg ** 2 / 16) - J0

            group_weights = weights[indices.to(device)]                                  # [n_group, 3, n_thetas]
            I0 = _complex_matmul_real(group_weights[:, 0], J0)[:, r_indices]
            I1 = _complex_matmul_real(group_weights[:, 1], J1)[:, r_indices]
            I2 = _complex_matmul_real(group_weights[:, 2], J2)[:, r_indices]

            varphi = torch.atan2(dy, dx).to(device)
            cos_phi, sin_phi = torch.cos(varphi), torch.sin(varphi)
            cos_2phi, sin_2phi = cos_phi ** 2 - sin_phi ** 2, 2.0 * sin_phi * cos_phi
            field_x = -1j * k_i * (0.5 * p_x * (I0 - I2 * cos_2phi) - 0.5 * p_y * I2 * sin_2phi
                                   + 1j * p_z * I1 * cos_phi)
            field_y = -1j * k_i * (-0.5 * p_x * I2 * sin_2phi + 0.5 * p_y * (I0 + I2 * cos_2phi)
                                   + 1j * p_z * I1 * sin_phi)
            image[indices.to(device)] = torch.stack([field_x, field_y], dim=1)
        return image
