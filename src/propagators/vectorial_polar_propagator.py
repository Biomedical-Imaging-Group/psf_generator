import math

import torch
from functorch import vmap
from torch.special import bessel_j0, bessel_j1

from integrators import simpsons_rule
from .vectorial_propagator import VectorialPropagator
from .polar_propagator import PolarPropagator


class VectorialPolarPropagator(VectorialPropagator, PolarPropagator):

    def _get_input_field(self) -> torch.Tensor:
        """Get the input field for vectorial polar propagator."""
        return self.pupil.field

    def compute_focus_field(self):
        # multiplicative scalar factor to be verified for the vectorial case
        self.field = self._compute_psf_for_far_field(self.pupil.field)
        return self.field

    def _compute_psf_for_far_field(self, far_fields):
        sin_t = torch.sin(self.thetas)  # [n_thetas, ]
        cos_t = torch.cos(self.thetas)  # [n_thetas, ]

        # bessel function evaluations are expensive and can be computed independently from defocus
        bessel_arg = self.k * self.rs[None, :] * sin_t[:, None]
        J0s = bessel_j0(bessel_arg)  # [n_theta, n_radii]
        J1s = bessel_j1(bessel_arg)  # [n_theta, n_radii]
        # bessel_j2() evaluations expressed in terms of j0(), j1()
        J2s = 2.0 * torch.where(bessel_arg > 1e-6,
                                J1s / bessel_arg,
                                0.5 - bessel_arg ** 2 / 16) - J0s

        # compute PSF field; handle defocus via batching with vmap()
        batched_compute_field_at_defocus = vmap(self._compute_psf_at_defocus,
                                                in_dims=(0, None, None, None, None, None, None))
        fields = batched_compute_field_at_defocus(self.defocus_filters, J0s, J1s, J2s, far_fields, sin_t, cos_t)
        return fields

    def _compute_psf_at_defocus(self, defocus_term, J0s, J1s, J2s, far_fields, sin_t, cos_t):
        field_x, field_y = far_fields[:, 0, :].squeeze(), far_fields[:, 1, :].squeeze()

        # compute E(r) for a list of unique radii values
        # shape(Ix0) == shape(Iy0) == ... == [n_radii,]
        # TODO: extend `quadrature_rule` to act on vector-valued functions?
        I_term = sin_t * (cos_t + 1.0) * defocus_term * self.correction_factor
        Ix0 = self.quadrature_rule(dx=self.dtheta,
                                   fs=J0s * (field_x * I_term)[:, None])
        Iy0 = self.quadrature_rule(dx=self.dtheta,
                                   fs=J0s * (field_y * I_term)[:, None])

        I_term = sin_t ** 2 * defocus_term * self.correction_factor
        Ix1 = self.quadrature_rule(dx=self.dtheta,
                                   fs=J1s * (field_x * I_term)[:, None])
        Iy1 = self.quadrature_rule(dx=self.dtheta,
                                   fs=J1s * (field_y * I_term)[:, None])

        I_term = sin_t * (cos_t - 1.0) * defocus_term * self.correction_factor
        Ix2 = self.quadrature_rule(dx=self.dtheta,
                                   fs=J2s * (field_x * I_term)[:, None])
        Iy2 = self.quadrature_rule(dx=self.dtheta,
                                   fs=J2s * (field_y * I_term)[:, None])

        # scatter the radial evaluations of E(r) onto the xy image grid
        # shape(Ix0) == shape(Iy0) == ... == [nx,ny]
        Ix0 = Ix0[self.rr_indices]
        Iy0 = Iy0[self.rr_indices]
        Ix1 = Ix1[self.rr_indices]
        Iy1 = Iy1[self.rr_indices]
        Ix2 = Ix2[self.rr_indices]
        Iy2 = Iy2[self.rr_indices]

        # PSF varphi coordinate
        varphi = torch.atan2(self.yy, self.xx)
        sin_phi = torch.sin(varphi)
        cos_phi = torch.cos(varphi)
        sin_twophi = 2.0 * sin_phi * cos_phi
        cos_twophi = cos_phi ** 2 - sin_phi ** 2
        self.sin_phi = sin_phi.to(self.device)
        self.cos_phi = cos_phi.to(self.device)
        self.sin_twophi = sin_twophi.to(self.device)
        self.cos_twophi = cos_twophi.to(self.device)

        # updated expression with correct 1j factors
        PSF_field = torch.stack([  # [n_channels=3, size_x, size_y]
            Ix0 - Ix2 * self.cos_twophi - Iy2 * self.sin_twophi,
            Iy0 - Ix2 * self.sin_twophi + Iy2 * self.cos_twophi,
            -2j * (Ix1 * self.cos_phi + Iy1 * self.sin_phi)],
            dim=0)

        return PSF_field / 2 / math.sqrt(self.refractive_index)