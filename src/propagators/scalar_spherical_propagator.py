import math

import torch
from functorch import vmap
from torch.special import bessel_j0

from .spherical_propagator import SphericalPropagator


class ScalarSphericalPropagator(SphericalPropagator):

    def get_input_field(self) -> torch.Tensor:
        """
        Define a (1D) radial pupil function for the scalar polar case. The function is defined on
        the interval `\rho` \in [0,1]; `\rho` is a "normalized" radius. The conversion to physical
        pupil coordinates - the polar angle `\theta` - is given by:

            \rho = \frac{\sin{\theta}}{\sin{\theta_{max}}}

        such that the physical domain is:

            \theta \leq \theta_{max}

        Get the input field for scalar polar propagator.
        Suqeezing is applied to get rid of the empty dimensions.
        After squeezing, the shape of pupil.field changes from [n_defocus=1, channels=1, n_thetas] to [n_thetas, ]
        """
        input_field = torch.ones(self.n_pix_pupil).to(torch.complex64).to(self.device).unsqueeze(0).unsqueeze(0)
        return (input_field * self._zernike_aberrations()).squeeze()


    def compute_focus_field(self) -> torch.Tensor:
        """Compute the focus field for scalar polar propagator.
        This invovles expensive evaluations of Bessel functions.
        We compute it independently from defocus and handle defocus via batching with vmap().

        Parameters:
        -----------
        self.thetas: torch.Tensor
            shape: (n_thetas, )
        self.dtheta: torch.Tensor[float]
        self.rs: torch.Tensor
            shape: (n_radii, )
        self.correction_factor: torch.Tensor
            shape: (n_thetas, )

        J0: torch.Tensor
            shape: (n_theta, n_radii)
        Returns
        -------
        field: torch.Tensor
            output field
        """
        input_field = self.get_input_field()

        sin_t = torch.sin(self.thetas)
        bessel_arg = self.k * self.rs[None, :] * sin_t[:, None]
        J0 = bessel_j0(bessel_arg)

        batched_compute_field_at_defocus = vmap(self._compute_psf_at_defocus, in_dims=(0, None, None, None))
        return batched_compute_field_at_defocus(self.defocus_filters, J0, input_field, sin_t)


    def _compute_psf_at_defocus(self, defocus_term, J0, input_field, sin_t) -> torch.Tensor:
        """Compute PSF at defocus.
        We first compute E(r)--`integrand` for a list of unique radii values, then scatter the radial evaluations
        of E(r) onto the xy image grid.

        Parameters
        ----------
        defocus_term
        J0: torch.Tensor
            Bessel function J0
        input_field: torch.Tensor
            input pupil field
        sin_t: torch.Tensor
            shape: (n_thetas, )

        Returns
        -------
        field: torch.Tensor
            output field at defocus, shape: (n_channels=1, size_x, size_y)
        """
        integrand = J0 * (input_field * defocus_term * self.correction_factor * sin_t)[:, None]  # [n_theta, n_radii]
        field = self.quadrature_rule(fs=integrand, dx=self.dtheta)
        field = field[self.rr_indices].unsqueeze(0)
        return field / math.sqrt(self.refractive_index)