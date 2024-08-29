import math

import torch
from functorch import vmap
from torch.special import bessel_j0

from .polar_propagator import PolarPropagator
from .scalar_propagator import ScalarPropagator


class ScalarPolarPropagator(ScalarPropagator, PolarPropagator):
    def compute_focus_field(self):
        far_fields = self.pupil.field.squeeze()   # [n_defocus=1, channels=1, n_thetas] ==> [n_thetas, ]
        return self._compute_psf_for_far_field(far_fields)

    def _compute_psf_for_far_field(self, far_fields):
        # argument shapes:
        # self.thetas,            [n_thetas, ]
        # self.dtheta,            float
        # self.rs,                [n_radii, ]
        # self.correction_factor  [n_thetas, ]
        # far_fields              [n_thetas, ]
        sin_t = torch.sin(self.thetas) # [n_thetas, ]

        # bessel function evaluations are expensive and can be computed independently from defocus
        J_evals = bessel_j0(self.k * self.rs[None,:] * sin_t[:,None])    # [n_theta, n_radii]

        # compute PSF field; handle defocus via batching with vmap()
        batched_compute_field_at_defocus = vmap(self._compute_psf_at_defocus, in_dims=(0, None, None, None))

        fields = batched_compute_field_at_defocus(self.defocus_filters, J_evals, far_fields, sin_t)
        return fields

    def _compute_psf_at_defocus(self, defocus_term, J_evals, far_fields, sin_t):
        # compute E(r) for a list of unique radii values
        integrand = J_evals * (far_fields * defocus_term * self.correction_factor * sin_t)[:,None]  # [n_theta, n_radii]
        field = self.quadrature_rule(integrand, self.dtheta)
        # scatter the radial evaluations of E(r) onto the xy image grid
        field = field[self.rr_indices].unsqueeze(0)       # [n_channels=1, size_x, size_y]
        return field / math.sqrt(self.refractive_index)