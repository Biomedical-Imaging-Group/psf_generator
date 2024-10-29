# Copyright Biomedical Imaging Group, EPFL 2024

"""
The abstract propagator class.

"""
from abc import ABC, abstractmethod

import torch


class Propagator(ABC):
    r"""
    TODO: add description
    """
    def __init__(self,
                 n_pix_pupil: int =128,
                 n_pix_psf: int = 128,
                 device: str = 'cpu',
                 zernike_coefficients=None,
                 wavelength: int = 632,
                 na: float = 1.3,
                 fov: int = 2000,
                 refractive_index: float = 1.5,
                 defocus_min: float = 0.0,
                 defocus_max: float = 0.0,
                 n_defocus: int = 1,
                 apod_factor: bool = False,
                 envelope=None,
                 gibson_lanni: bool = False,
                 z_p: float = 1e3,
                 n_s: float = 1.3,
                 n_g: float = 1.5,
                 n_g0: float = 1.5,
                 t_g: float = 170e3,
                 t_g0: float = 170e3,
                 n_i: float = 1.5,
                 t_i0: float = 100e3):
        # number of pixels (size) of the pupil, assuming square image
        self.n_pix_pupil = n_pix_pupil
        # number of pixels (size) of the psf, assuming square image
        self.n_pix_psf = n_pix_psf
        # device: "cpu" or "gpu"
        self.device = device
        # Zernike coefficients
        if zernike_coefficients is None:
            zernike_coefficients = [0]
        if not isinstance(zernike_coefficients, torch.Tensor):
            zernike_coefficients = torch.tensor(zernike_coefficients)
        self.zernike_coefficients = zernike_coefficients
        # All distances are in nanometers
        # wavelength
        self.wavelength = wavelength
        # numerical aperture
        self.na = na
        # size of the field-of-view
        self.fov = fov
        # refractive index
        self.refractive_index = refractive_index
        # minimal distance of defocus
        self.defocus_min = defocus_min
        # maximal distance of defocus
        self.defocus_max = defocus_max
        # number of steps for defocus
        self.n_defocus = n_defocus
        # whether to apply apodization factor
        self.apod_factor = apod_factor
        # envelope of the PSF intensity
        self.envelope = envelope
        # whether to apply Gibson-Lanni aberration
        self.gibson_lanni = gibson_lanni
        # constants in the Gibson-Lanni aberration formula
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

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Get name of the propagator in a certain format, e.g. 'scalar_cartesian'."""
        raise NotImplementedError

    @abstractmethod
    def _zernike_aberrations(self) -> torch.Tensor:
        """Zernike aberrations that will be applied on the pupil."""
        raise NotImplementedError

    @abstractmethod
    def get_input_field(self) -> torch.Tensor:
        """Get the corresponding pupil as the input field of propagator."""
        raise NotImplementedError

    @abstractmethod
    def compute_focus_field(self) -> torch.Tensor:
        """Compute the output field of the propagator at focal plane."""
        raise NotImplementedError

    def compute_optical_path(self, sin_t: torch.Tensor) -> torch.Tensor:
        """Compute the optical path following Eq. (3.45) in [1].

        References
        ----------
        .. [1] https://bigwww.epfl.ch/publications/aguet0903.pdf

        """
        path = self.z_p * torch.sqrt(self.n_s ** 2 - self.n_i ** 2 * sin_t ** 2) \
               + self.t_i * torch.sqrt(self.n_i ** 2 - self.n_i ** 2 * sin_t ** 2) \
               - self.t_i0 * torch.sqrt(self.n_i0 ** 2 - self.n_i ** 2 * sin_t ** 2) \
               + self.t_g * torch.sqrt(self.n_g ** 2 - self.n_i ** 2 * sin_t ** 2) \
               - self.t_g0 * torch.sqrt(self.n_g0 ** 2 - self.n_i ** 2 * sin_t ** 2)
        return path
