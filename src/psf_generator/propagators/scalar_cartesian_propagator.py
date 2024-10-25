# Copyright Biomedical Imaging Group, EPFL 2024

"""
The propagator for scalar field in Cartesian coordinates.

"""

import torch

from .cartesian_propagator import CartesianPropagator
from ..utils.zernike import create_pupil_mesh


class ScalarCartesianPropagator(CartesianPropagator):
    r"""
    TODO: add description and formulae
    """

    @classmethod
    def get_name(cls) -> str:
        return 'scalar_cartesian'

    def get_input_field(self) -> torch.Tensor:
        r"""
        Define the corresponding 2D pupil function as the input field.

        Notes
        -----
        This function is defined on the unit disk centered at (0,0)

        .. math:: u^2 + v^2 <= 1.

        The mapping between this domain and the physical pupil coordinates are

        .. math::

            u = s_x / s_{\mathrm{max}},
            v = s_y / s_{\mathrm{max}}.

        such that the physical domain is:

        .. math:: s_x^2 + s_y^2 <= s_{\mathrm{max}}^2 = \sin(\theta_{\mathrm{max}})^2.

        """
        kx, ky = create_pupil_mesh(n_pixels=self.n_pix_pupil)
        input_field = (kx**2 + ky**2 <= 1).to(torch.complex64).to(self.device)
        return input_field * self._zernike_aberrations()


