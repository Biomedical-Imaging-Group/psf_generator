# Copyright Biomedical Imaging Group, EPFL 2024

"""
The abstract propagator class.

"""
import json
import os
from abc import ABC, abstractmethod

import torch

from ..utils.misc import convert_tensor_to_array


class Propagator(ABC):
    r"""
    Base class propagator.

    Parameters
    ----------
    n_pix_pupil : int
        Number of pixels (size) of the pupil (always a square image).
    n_pix_psf : int
        Number of pixels (size) of the PSF (always a square image).
    device : str
        Computational backend. Choose from 'cpu' and 'gpu'.
    zernike_coefficients : np.ndarray or torch.tensor
        Zernike coefficients of length 'K' of the chosen first 'K' modes.
    wavelength : float
        Wavelength of light, in nanometer.
    na : float
        Numerical aperture.
    fov : float
        Size of the square field of view of the PSF plane, in micrometer.
    defocus_min : float
        Extent of the defocus along the optical (z) axis on one side of the focal plane in micrometer.
    defocus_max : float
        Extent of the defocus along the optical (z) axis on the other side of the focal plane in micrometer.
    n_defocus : int
        Number of z-stack.
    apod_factor : bool
        Apply apodization factor or not.
    envelope : float
        Size :math:`k_{\mathrm{env}}` of the Gaussian envolope :math:`A(\mathbf{s}) = \mathrm{e}^{-(k^2_x+k^2_y)/k_\mathrm{env}^2}`.
    gibson_lanni : bool
        Apply Gibson-Lanni aberration or not.
    z_p : float
        Depth of the focal plane in the sample. It is usually obtained experimentally by focusing on a point source at this depth. 
    n_s : float
        Refractive index of the sample.
    n_g : float
        Refractive index of the (glass) cover slip.
    n_g0 : float
        Design condition of the refractive index of the cover slip.
    t_g : float
        Thickness of the sample.
    t_g0 : float
        Design condition of the thickness of the sample.
    n_i : float
        Refractive index of the immersion medium.
    n_i0 : float
        Design condition of the refractive index of the immersion medium.
    t_i : float
        Thickness of the immersion medium. It is computed from
        :math:`t_i = z_p - z + n_i \left( -\frac{z_p}{n_s} - \frac{t_g}{n_g} + \frac{t_g^0}{n_g^0} + \frac{t_i^0}{n_i^0} \right)`.
    t_i0 : float
        Design condition of the thickness of the immersion medium.
    refractive_index : float
        Refractive index of the propagation medium, i.e. sample if gibson_lanni=True, immersion oil otherwise.

    Notes
    -----
    `(z_p, n_s, n_g, n_g0, t_g, t_g0, n_i, t_i0, t_i)` are coefficients related to the aberrations due to refractive
    index mismatch between stratified layers of the microscope.
    This aberration is computed by method `self.compute_optical_path`.

    """
    def __init__(self,
                 n_pix_pupil: int =128,
                 n_pix_psf: int = 128,
                 device: str = 'cpu',
                 zernike_coefficients=None,
                 wavelength: float = 632,
                 na: float = 1.3,
                 fov: float = 2000,
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
                 n_i0: float = 1.5,
                 t_i0: float = 100e3):
        self.n_pix_pupil = n_pix_pupil
        self.n_pix_psf = n_pix_psf
        self.device = device
        if zernike_coefficients is None:
            zernike_coefficients = [0]
        if not isinstance(zernike_coefficients, torch.Tensor):
            zernike_coefficients = torch.tensor(zernike_coefficients)
        self.zernike_coefficients = zernike_coefficients
        self.wavelength = wavelength
        self.na = na
        self.fov = fov
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
        self.n_i0 = n_i0
        self.t_i0 = t_i0
        self.t_i = n_i * (t_g0 / n_g0 + t_i0 / self.n_i0 - t_g / n_g - z_p / n_s)
        if gibson_lanni:
            self.refractive_index = n_s
        else:
            self.refractive_index = 1.0

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Get name of the propagator in a certain format, e.g. 'scalar_cartesian'."""
        raise NotImplementedError

    @abstractmethod
    def _aberrations(self) -> torch.Tensor:
        """Aberrations that will be applied on the pupil."""
        raise NotImplementedError

    @abstractmethod
    def get_input_field(self) -> torch.Tensor:
        """Get the input field of propagator."""
        raise NotImplementedError

    @abstractmethod
    def compute_focus_field(self) -> torch.Tensor:
        """Compute the output field of the propagator at focal plane."""
        raise NotImplementedError

    def get_pupil(self) -> torch.Tensor:
        """Get the pupil function."""
        return self.get_input_field() * self._aberrations()

    def compute_optical_path(self, sin_t: torch.Tensor) -> torch.Tensor:
        r"""Compute the optical path following Eq. (3.45) in [1]

        .. math::

                W(\mathbf{s}) &=
                 k \left( t_s \sqrt{n_s^2 - n_i^2 \sin^2 \theta}
                 + t_i \sqrt{n_i^2 - n_i^2 \sin^2 \theta}
                 -t_i^* \sqrt{\left.n_i^*\right.^2 - n_i^2 \sin^2 \theta} \right. \\
                & \quad \left. + t_g \sqrt{n_g^2 - n_i^2 \sin^2 \theta}
                - t_g^* \sqrt{\left.n_g^*\right.^2 - n_i^2 \sin^2 \theta}\right).


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

    def _get_args(self) -> dict:
        """Get the parameters of the propagator."""
        args = {
            'n_pix_pupil': self.n_pix_pupil,
            'n_pix_psf': self.n_pix_psf,
            'device': self.device,
            'zernike_coefficients': convert_tensor_to_array(self.zernike_coefficients).tolist(),
            'wavelength': self.wavelength,
            'na': self.na,
            'fov': self.fov,
            'refractive_index': self.refractive_index,
            'defocus_min': self.defocus_min,
            'defocus_max': self.defocus_max,
            'n_defocus': self.n_defocus,
            'apod_factor': self.apod_factor,
            'envelope': self.envelope,
            'gibson_lanni': self.gibson_lanni,
            'z_p': self.z_p,
            'n_s': self.n_s,
            'n_g': self.n_g,
            'n_g0': self.n_g0,
            't_g': self.t_g,
            't_g0': self.t_g0,
            'n_i': self.n_i,
            't_i0': self.t_i0,
            't_i': self.t_i
        }
        return args

    def save_parameters(self, json_filepath: str):
        r"""
        Save the parameters of the propagator in a JSON file.

        Notes
        -----
        - Zernike coefficients are converted to a list
        - complex numbers, e.g. e0x or e0y, are converted to a string

        Parameters
        ----------
        json_filepath : str, optional
            Path to save the attributes in a JSON file.

        """
        args = self._get_args()
        os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
        with open(json_filepath, 'w') as file:
            json.dump(args, file, indent=2)

