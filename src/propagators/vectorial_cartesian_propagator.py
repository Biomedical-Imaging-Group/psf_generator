import torch

from .cartesian_propagator import CartesianPropagator
from .vectorial_propagator import VectorialPropagator
from utils.czt import custom_ifft2


class VectorialCartesianPropagator(VectorialPropagator, CartesianPropagator):

    def _get_input_field(self):
        # Angles theta and phi
        sin_xx, sin_yy = torch.meshgrid(self.s_x * self.s_max, self.s_x * self.s_max, indexing='ij')
        sin_t_sq = sin_xx ** 2 + sin_yy ** 2
        s_valid = torch.tensor(sin_t_sq <= self.s_max ** 2)
        sin_theta = torch.sqrt(sin_t_sq)
        cos_theta = torch.sqrt(1.0 - sin_t_sq)
        phi = torch.atan2(sin_yy, sin_xx)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        sin_2phi = 2.0 * sin_phi * cos_phi
        cos_2phi = cos_phi ** 2 - sin_phi ** 2

        # Field after basis change
        field_x, field_y = self.pupil.field[:, 0, :, :], self.pupil.field[:, 1, :, :]
        e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * field_x \
                  + (cos_theta - 1.0) * sin_2phi * field_y
        e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * field_y \
                  + (cos_theta - 1.0) * sin_2phi * field_x
        e_inf_z = -2.0 * sin_theta * (cos_phi * field_x + sin_phi * field_y)

        e_infs = [torch.where(s_valid, e_inf, 0.0).unsqueeze(0) / 2
                  for e_inf in (e_inf_x, e_inf_y, e_inf_z)]
        e_inf_field = torch.cat(e_infs, dim=1)
        return e_inf_field
