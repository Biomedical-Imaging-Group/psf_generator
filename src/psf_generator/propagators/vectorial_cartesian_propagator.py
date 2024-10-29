# Copyright Biomedical Imaging Group, EPFL 2024

"""
The propagator for the vectorial field in the Cartesian coordinates.
"""

import torch

from .cartesian_propagator import CartesianPropagator
from ..utils.zernike import create_pupil_mesh


class VectorialCartesianPropagator(CartesianPropagator):
    """
    TODO: add description and formulae
    """
    def __init__(self, n_pix_pupil=128, n_pix_psf=128, device='cpu',
                 zernike_coefficients=None,
                 special_phase_mask=None,
                 e0x=1.0, e0y=0.0,
                 wavelength=632, na=1.3, fov=1000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, t_i0=100e3):
        super().__init__(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix_psf, device=device,
                         zernike_coefficients=zernike_coefficients,
                         special_phase_mask=special_phase_mask,
                         wavelength=wavelength, na=na, fov=fov, refractive_index=refractive_index,
                         defocus_min=defocus_min, defocus_max=defocus_max, n_defocus=n_defocus,
                         apod_factor=apod_factor, envelope=envelope,
                         gibson_lanni=gibson_lanni, z_p=z_p, n_s=n_s,
                         n_g=n_g, n_g0=n_g0, t_g=t_g, t_g0=t_g0,
                         n_i=n_i, t_i0=t_i0)

        # electric field component ex at focal plane
        self.e0x = e0x
        # electric field component ey at focal plane
        self.e0y = e0y

    @classmethod
    def get_name(cls) -> str:
        return 'vectorial_cartesian'

    def get_input_field(self) -> torch.Tensor:
        r"""
        Compute the corresponding input field.

        TODO: more explanations. Use :math:`\pi` for math formulae.

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
                           dim=0).to(self.device) * self._zernike_aberrations()

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
