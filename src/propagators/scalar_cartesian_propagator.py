import torch

from .cartesian_propagator import CartesianPropagator
from .scalar_propagator import ScalarPropagator
from utils.czt import custom_ifft2


class ScalarCartesianPropagator(ScalarPropagator, CartesianPropagator):
    def compute_focus_field(self):
        self.field = self._compute_psf_for_far_field(self.pupil.field)
        return self.field

    def _compute_psf_for_far_field(self, far_fields):
        k_start = -self.zoom_factor * torch.pi
        k_end   =  self.zoom_factor * torch.pi
        field = custom_ifft2(far_fields * self.correction_factor * self.defocus_filters,
                                  shape_out=(self.n_pix_psf, self.n_pix_psf),
                                  k_start=k_start,
                                  k_end=k_end,
                                  norm='forward', fftshift_input=True, include_end=True) \
                                      * (self.ds * self.s_max) ** 2
        return field / (2 * torch.pi * torch.sqrt(torch.tensor(self.refractive_index)))

