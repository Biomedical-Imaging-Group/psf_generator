from abc import ABC, abstractmethod

import torch


class Propagator(ABC):
    def __init__(self, pupil, n_pix_psf=128, device='cpu',
                 wavelength=632, na=1.3, fov=2000, refractive_index=1.5,
                 defocus_min=0, defocus_max=0, n_defocus=1,
                 apod_factor=False, envelope=None,
                 gibson_lanni=False, z_p=1e3, n_s=1.3,
                 n_g=1.5, n_g0=1.5, t_g=170e3, t_g0=170e3,
                 n_i=1.5, t_i0=100e3):
        self.pupil = pupil

        self.n_pix_psf = n_pix_psf
        self.n_pix_pupil = pupil.n_pix_pupil
        self.device = device
        if self.device != pupil.device:
            print('Warning: device of propagator and pupil are not the same.')
            print('Pupil device: ', pupil.device)
            print('Propagator device: ', self.device)
            print('Setting propagator device to pupil device.')
            self.device = pupil.device

        # All distances are in nanometers
        self.wavelength = wavelength
        self.na = na
        self.fov = fov
        self.refractive_index = refractive_index

        self.defocus_min = defocus_min
        self.defocus_max = defocus_max
        self.n_defocus = n_defocus

        self.apod_factor = apod_factor
        self.envelope = envelope

        self.gibson_lanni = gibson_lanni
        self.z_p = z_p
        self.n_s = n_s
        self.n_g = n_g
        self.n_g0 = n_g0
        self.t_g = t_g
        self.t_g0 = t_g0
        self.n_i = n_i
        self.n_i0 = refractive_index
        self.t_i0 = t_i0
        self.t_i = n_i * (t_g0 / n_g0 + t_i0 / self.n_i0 - t_g / n_g - z_p / n_s)

        self.field = None

    @abstractmethod
    def _get_input_field(self):
        """Get the corresponding pupil as the input field for propagator."""
        raise NotImplementedError

    @abstractmethod
    def compute_focus_field(self):
        """Compute the focus field - PSF - output of hte propagator."""
        raise NotImplementedError

    def compute_optical_path(self, sin_t: torch.Tensor) -> torch.Tensor:
        """Compute the optical path following Eq. (3.45) of François Aguet's thesis."""
        path = self.z_p * torch.sqrt(self.n_s ** 2 - self.n_i ** 2 * sin_t ** 2) \
               + self.t_i * torch.sqrt(self.n_i ** 2 - self.n_i ** 2 * sin_t ** 2) \
               - self.t_i0 * torch.sqrt(self.n_i0 ** 2 - self.n_i ** 2 * sin_t ** 2) \
               + self.t_g * torch.sqrt(self.n_g ** 2 - self.n_i ** 2 * sin_t ** 2) \
               - self.t_g0 * torch.sqrt(self.n_g0 ** 2 - self.n_i ** 2 * sin_t ** 2)
        return path
