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

    def _compute_psf_for_far_field(self, far_fields):  # to remove later?
        s_xx, s_yy = torch.meshgrid(self.s_x * self.s_max, self.s_x * self.s_max, indexing='ij')
        sin_t_sq = s_xx ** 2 + s_yy ** 2
        s_valid = torch.Tensor(sin_t_sq <= self.s_max ** 2)
        sin_theta = torch.sqrt(sin_t_sq)
        cos_theta = torch.sqrt(1.0 - sin_t_sq)
        phi = torch.atan2(s_yy, s_xx)   # properly handles pole at sin_theta == 0.0
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        sin_2phi = 2.0 * sin_phi * cos_phi
        cos_2phi = cos_phi ** 2 - sin_phi ** 2

        far_fields_x = far_fields[:, 0, :, :]
        far_fields_y = far_fields[:, 1, :, :]
        e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * far_fields_x \
                        + (cos_theta - 1.0) * sin_2phi * far_fields_y
        e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * far_fields_y \
                        + (cos_theta - 1.0) * sin_2phi * far_fields_x
        e_inf_z = -2.0 * sin_theta * (cos_phi * far_fields_x + sin_phi * far_fields_y)

        e_inf_x = torch.where(s_valid, e_inf_x, 0.0).unsqueeze(0)
        e_inf_y = torch.where(s_valid, e_inf_y, 0.0).unsqueeze(0)
        e_inf_z = torch.where(s_valid, e_inf_z, 0.0).unsqueeze(0)

        PSF_field = custom_ifft2(torch.cat((e_inf_x, e_inf_y, e_inf_z), dim=1) * self.correction_factor * self.defocus_filters,
                                  shape_out=(self.n_pix_psf, self.n_pix_psf),
                                  k_start=-self.zoom_factor * torch.pi,
                                  k_end=self.zoom_factor * torch.pi,
                                  norm='forward', fftshift_input=True, include_end=True) \
                     * (self.ds * self.s_max) ** 2 * 1j
        PSF_field /= (2 * torch.pi * torch.sqrt(torch.tensor(self.refractive_index)))

        return PSF_field
