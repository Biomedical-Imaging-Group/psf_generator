# Copyright Biomedical Imaging Group, EPFL 2024

"""
The propagator for scalar field in Spherical coordinates.

"""

import math

import torch
from functorch import vmap
from torch.special import bessel_j0

from .spherical_propagator import SphericalPropagator


class ScalarSphericalPropagator(SphericalPropagator):

    @classmethod
    def get_name(cls) -> str:
        return 'scalar_spherical'

    def get_input_field(self) -> torch.Tensor:
        """
        Define a (1D) radial pupil function as the input field.

        Notes
        -----
        This function is defined on the interval :math:`\rho \in [0,1]`; :math:`\rho` is a "normalized" radius.
        The conversion to physical pupil coordinates - the polar angle :math:`\theta` - is given by

        .. `math:`\rho = \frac{\sin{\theta}}{\sin{\theta_{max}}}`,

        such that the physical domain is

       .. :math:`\theta \leq \theta_{max}`.

        TODO:
        Suqeezing is applied to get rid of the empty dimensions.
        After squeezing, the shape of pupil.field changes from [n_defocus=1, channels=1, n_thetas] to [n_thetas, ]

        """
        input_field = torch.ones(self.n_pix_pupil).to(torch.complex64).to(self.device).unsqueeze(0).unsqueeze(0)
        return (input_field * self._zernike_aberrations()).squeeze()


    def compute_focus_field(self) -> torch.Tensor:
        """Compute the focus field for scalar spherical propagator.

        Parameters:
        -----------
        self.thetas : torch.Tensor
            List of angle of shape: (n_thetas, )
        self.rs : torch.Tensor
            TODO: what is it? of shape: (n_radii, )
        self.correction_factor : torch.Tensor
            Correction factor of shape: (n_thetas, )
        J0 : torch.Tensor
            Bessel function J0 of shape: (n_theta, n_radii)

        Returns
        -------
        field: torch.Tensor
            output field

        Notes
        -----
        This involves expensive evaluations of Bessel functions.
        We compute it independently of defocus and handle defocus via batching with vmap().

        """
        input_field = self.get_input_field()

        sin_t = torch.sin(self.thetas)
        bessel_arg = self.k * self.rs[None, :] * sin_t[:, None]
        J0 = bessel_j0(bessel_arg)

        batched_compute_field_at_defocus = vmap(self._compute_psf_at_defocus, in_dims=(0, None, None, None))
        return batched_compute_field_at_defocus(self.defocus_filters, J0, input_field, sin_t)


    def _compute_psf_at_defocus(
            self,
            defocus_term,
            J0: torch.Tensor,
            input_field: torch.Tensor,
            sin_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute PSF at defocus.

        Parameters
        ----------
        defocus_term :
            TODO: what is it?
        J0 : torch.Tensor
            Bessel function J0
        input_field : torch.Tensor
            input pupil field
        sin_t : torch.Tensor
            TODO: what is it? of shape: (n_thetas, )

        Returns
        -------
        field: torch.Tensor
            output field at defocus of shape: (n_channels=1, size_x, size_y)

        Notes
        -----
        We first compute E(r)--`integrand` for a list of unique radii values, then scatter the radial evaluations
        of E(r) onto the xy image grid.

        """
        integrand = J0 * (input_field * defocus_term * self.correction_factor * sin_t)[:, None]  # [n_theta, n_radii]
        field = self.quadrature_rule(fs=integrand, dx=self.dtheta)
        field = field[self.rr_indices].unsqueeze(0)
        return field / math.sqrt(self.refractive_index)