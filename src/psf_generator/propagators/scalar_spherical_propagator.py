# Copyright Biomedical Imaging Group, EPFL 2025

"""
The propagator for scalar field in Spherical coordinates.

"""

import math

import torch
from torch import vmap
from torch.special import bessel_j0

from .spherical_propagator import SphericalPropagator


class ScalarSphericalPropagator(SphericalPropagator):
    r"""
    Propagator for the scalar approximation of the Richard's-Wolf integral in spherical parameterization.

    The equation to compute the eletric field is

    .. math::
            E(\boldsymbol{\rho})
                = -\mathrm{i}fk \int_0^{\theta_{\max}} d\theta \mathrm{e}_{\infty}(\theta)
                J_0(k \rho \sin \theta) \mathrm{e}^{\mathrm{i} kz\cos\theta} \sin\theta,

    where :math:`J_0` is the Bessel function of first kind and order 0.

    """

    @classmethod
    def get_name(cls) -> str:
        return 'scalar_spherical'

    def get_input_field(self) -> torch.Tensor:
        r"""
        Define a (1D) radial pupil function as the input field.

        Notes
        -----
        This function is defined on the interval :math:`\rho \in [0,1]`; :math:`\rho` is a "normalized" radius.
        The conversion to physical pupil coordinates - the polar angle :math:`\theta` - is given by

        .. math:: \rho = \frac{\sin{\theta}}{\sin{\theta_{\max}}},

        such that the physical domain is

        .. math:: \theta \leq \theta_{\max}.

        """
        input_field = torch.ones(self.n_pix_pupil).to(torch.complex64).to(self.device)
        return input_field


    def compute_focus_field(self) -> torch.Tensor:
        r"""Compute the focus field for scalar spherical propagator.

        Parameters
        ----------
        self.thetas : torch.Tensor
            Angles of sampling of shape `(n_thetas, )`.
        self.rs : torch.Tensor
            Radii of sampling of shape `(n_radii, )`.
        self.correction_factor : torch.Tensor
            Correction factor of shape `(n_thetas, )`.
        J0 : torch.Tensor
            Bessel function of the first kind of order 0 :math:`J_0`. Shape: `(n_theta, n_radii)`.

        Returns
        -------
        field: torch.Tensor
            Output field.

        Notes
        -----
        This involves expensive evaluations of Bessel functions.
        We compute it independently of defocus and handle defocus via batching with vmap().

        """
        sin_t = torch.sin(self.thetas)
        bessel_arg = self.k * self.rs[None, :] * sin_t[:, None] * self.refractive_index
        J0 = bessel_j0(bessel_arg)

        batched_compute_field_at_defocus = vmap(self._compute_psf_at_defocus, in_dims=(0, None, None, None))
        return batched_compute_field_at_defocus(self.defocus_filters, J0, self.get_pupil(), sin_t)


    def _compute_psf_at_defocus(
            self,
            defocus_term,
            J0: torch.Tensor,
            pupil: torch.Tensor,
            sin_t: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute PSF at defocus.

        Parameters
        ----------
        defocus_term:
            Factor in the integrand corresponding to defocus.
        J0: torch.Tensor
            Bessel function of the first kind of order 0 :math:`J_0`.
        pupil: torch.Tensor
            Pupil function.
        sin_t: torch.Tensor
            Factor in the integrand of shape: `(n_thetas, )`.

        Returns
        -------
        field: torch.Tensor
            Output field at defocus. Shape: `(n_channels=1, size_x, size_y)`.

        Notes
        -----
        We first compute E(r)--`integrand` for a list of unique radii values, then scatter the radial evaluations
        of E(r) onto the xy image grid.

        """
        integrand = J0 * (pupil * defocus_term * sin_t)[:, None]  # [n_theta, n_radii]
        field = self.integrator(fs=integrand, dx=self.dtheta)
        field = field[self.rr_indices].unsqueeze(0)
        return field / math.sqrt(self.refractive_index)
