import torch

from utils.zernike import create_pupil_mesh
from .cartesian_propagator import CartesianPropagator


class ScalarCartesianPropagator(CartesianPropagator):

    def get_input_field(self):
        """
        Define a 2D pupil function for the scalar Cartesian case. The function is defined on the
        unit disk centered at (0,0): u ** 2 + v ** 2 <= 1. The mapping between this domain and
        the physical pupil coordinates are:

            u = sx / s_max
            v = sy / s_max

        such that the physical domain is:

            sx ** 2 + sy ** 2 <= s_max ** 2 = sin(theta_max) ** 2
        """
        kx, ky = create_pupil_mesh(n_pixels=self.n_pix_pupil)
        input_field = (kx**2 + ky**2 <= 1).to(torch.complex64).unsqueeze(0).unsqueeze(0).to(self.device)
        return input_field * self._zernike_aberrations()


