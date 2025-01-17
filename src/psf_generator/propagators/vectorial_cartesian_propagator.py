# Copyright Biomedical Imaging Group, EPFL 2025

"""
The propagator for the vectorial field in the Cartesian coordinates.
"""

import typing as tp

import torch

from .cartesian_propagator import CartesianPropagator
from ..utils.zernike import create_pupil_mesh


class VectorialCartesianPropagator(CartesianPropagator):
    r"""
    Propagator for the vectorial case of the Richard's-Wolf integral in Cartesian parameterization.

    In the vectorial model, the far field :math:`\boldsymbol{e}_{\infty}` depends on the vectorial incident field
    :math:`\boldsymbol{e}_{\textrm{inc}} = [\boldsymbol{e}_{\textrm{inc}}^x, \boldsymbol{e}_{\textrm{inc}}^y, 0]`
    as follows:

    .. math::
            \boldsymbol{e}_{\infty}(\theta,\phi) =
             \begin{bmatrix}
                (\cos\theta+1)+(\cos\theta-1)\cos2\phi \\
                (\cos\theta-1)\sin2\phi \\
                -2 \cos\phi \sin\theta
             \end{bmatrix} \frac{\boldsymbol{e}_{\textrm{inc}}^x}{2} +
             \begin{bmatrix}
                (\cos\theta-1)\sin2\phi  \\
                (\cos\theta+1)-(\cos\theta-1)\cos2\phi \\
                - 2 \sin\phi \sin\theta
             \end{bmatrix} \frac{\boldsymbol{e}_{\textrm{inc}}^y}{2}.

    The equation to compute the electric field is

    .. math::
            \mathbf{E}(\boldsymbol{\rho})
            = -\frac{\mathrm{i} fk}{2\pi}\iint\limits_{s_x^2+s_y^2 \leq s_{M}^2}
            \frac{\boldsymbol{e}_{\infty}(s_x, s_y) \mathrm{e}^{\mathrm{i} kz}}{s_z}
            \mathrm{e}^{\mathrm{i} k(s_x x + s_y y)} ds_x ds_y.

    Parameters
    ----------
    `self.e0x` : float, optional
        Initial electric field component :math:`\mathbf{e}_0^x`. Default value is `1.0`.
    `self.e0y` : float, optional
        Initial electric field component :math:`\mathbf{e}_0^y`. Default value is `0.0`.

    Notes
    -----
    The vectorial propagators have two additional arguments apart from those inherited form the base propagator
    to account for polarization.

    """

    def __init__(self, n_pix_pupil=128, n_pix_psf=128, device='cpu',
                 zernike_coefficients=None,
                 special_phase_mask=None,
                 e0x=1.0, e0y=0.0,
                 wavelength=632, na=1.3, fov=1000,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, n_i0=1.5, t_i0=100e3):
        super().__init__(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix_psf, device=device,
                         zernike_coefficients=zernike_coefficients,
                         special_phase_mask=special_phase_mask,
                         wavelength=wavelength, na=na, fov=fov,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, n_i0=n_i0, t_i0=t_i0)

        # electric field component ex at focal plane
        self.e0x = e0x
        # electric field component ey at focal plane
        self.e0y = e0y

    @classmethod
    def get_name(cls) -> str:
        return 'vectorial_cartesian'

    def _get_args(self) -> tp.Dict:
        args = super()._get_args()
        args['e0x'] = str(self.e0x)
        args['e0y'] = str(self.e0y)
        return args

    def get_input_field(self) -> torch.Tensor:
        r"""
        Compute the corresponding input field.
        """
        # Angles theta and phi
        sin_xx, sin_yy = torch.meshgrid(self.s_x * self.s_max, self.s_x * self.s_max, indexing='ij')
        sin_t_sq = sin_xx ** 2 + sin_yy ** 2
        s_valid = sin_t_sq <= self.s_max ** 2
        sin_theta = torch.sqrt(sin_t_sq)
        cos_theta = torch.sqrt(1.0 - sin_t_sq)
        phi = torch.atan2(sin_yy, sin_xx)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        sin_2phi = 2.0 * sin_phi * cos_phi
        cos_2phi = cos_phi ** 2 - sin_phi ** 2

        # Field after basis change
        kx, ky = create_pupil_mesh(n_pixels=self.n_pix_pupil)
        single_field = (kx ** 2 + ky ** 2 <= 1).to(torch.complex64)
        input_field = torch.stack((self.e0x * single_field, self.e0y * single_field),
                           dim=0).to(self.device)

        field_x, field_y = input_field[0, :, :], input_field[1, :, :]
        e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * field_x \
                  + (cos_theta - 1.0) * sin_2phi * field_y
        e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * field_y \
                  + (cos_theta - 1.0) * sin_2phi * field_x
        e_inf_z = -2.0 * sin_theta * (cos_phi * field_x + sin_phi * field_y)

        e_infs = [torch.where(s_valid, e_inf, 0.0) / 2
                  for e_inf in (e_inf_x, e_inf_y, e_inf_z)]
        e_inf_field = torch.stack(e_infs, dim=0)
        return e_inf_field
