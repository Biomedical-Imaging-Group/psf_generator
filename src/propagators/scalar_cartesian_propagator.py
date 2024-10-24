# Copyright Biomedical Imaging Group, EPFL 2024

"""
The propagator for scalar field in Cartesian coordinates.

"""

import torch

from utils.zernike import create_pupil_mesh
from .cartesian_propagator import CartesianPropagator


class ScalarCartesianPropagator(CartesianPropagator):

    @classmethod
    def get_name(cls) -> str:
        return 'scalar_cartesian'

    def get_input_field(self) -> torch.Tensor:
        """
        Define the corresponding 2D pupil function as the input field.

        Notes
        -----
        This function is defined on the unit disk centered at (0,0)

        .. :math:`u ** 2 + v ** 2 <= 1`.

        The mapping between this domain and the physical pupil coordinates are

        .. :math:`u = sx / s_max`,
        .. :math:`v = sy / s_max`.

        such that the physical domain is:

        .. :math:`sx ** 2 + sy ** 2 <= s_max ** 2 = sin(theta_max) ** 2`.

        """
        kx, ky = create_pupil_mesh(n_pixels=self.n_pix_pupil)
        input_field = (kx**2 + ky**2 <= 1).to(torch.complex64).to(self.device)
        return input_field * self._zernike_aberrations()


