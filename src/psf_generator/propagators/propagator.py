# Copyright Biomedical Imaging Group, EPFL 2025

"""
The abstract propagator class.

"""
from abc import abstractmethod

import torch

from ..utils.misc import centred_grid as _centred_grid
from ..utils.misc import convert_tensor_to_array
from ..utils.parameters import Parametrized
from ..utils.parameters import validate_device as _validate_device
from ..utils.parameters import validate_number as _validate_number
from ..utils.zernike import create_zernike_aberrations, zernike_basis


# Keys written by versions <= 0.1.0 that are derived from other parameters and are not constructor arguments.
_DERIVED_KEYS = ('refractive_index', 't_i')


class Propagator(Parametrized):
    r"""
    Base class propagator.

    Parameters
    ----------
    n_pix_pupil : int, optional
        Number of pixels (size) of the pupil (always a square image). Default value is `128`.
    n_pix_psf : int, optional
        Number of pixels (size) of the PSF (always a square image). Default value is `128`.
    device : str or torch.device, optional
        Computational backend, given as anything :func:`torch.device` accepts, e.g. `'cpu'`, `'cuda'`,
        `'cuda:0'` or `'mps'`. Default value is `'cpu'`.
    zernike_coefficients : np.ndarray or torch.tensor, optional
        Zernike coefficients of length 'K' of the chosen first 'K' modes. Default is `None`.
    wavelength : float, optional
        Wavelength of light, in nanometer. Default value is `632`.
    na : float, optional
        Numerical aperture. Default value is `1.3`.
    pix_size : float, optional
        Camera pixel size, in nanometer. This is the sampling step of the PSF grid: the PSF is evaluated at
        :math:`x_i = (i - \lfloor n_{\mathrm{pix}}/2 \rfloor)\, \mathrm{pix\_size}` along both lateral axes, so
        the optical axis goes through pixel ``n_pix_psf // 2`` (see attribute ``x``). Default value is `20`.
    defocus_step : float, optional
        Distance between consecutive z-slices, in nanometer. The slices are located at
        :math:`z_i = (i - \lfloor n_{\mathrm{defocus}}/2 \rfloor)\, \mathrm{defocus\_step}`, so slice
        ``n_defocus // 2`` is the focal plane (see attribute ``z``). Default value is `0.0`.
    n_defocus : int, optional
        Number of z-slices. Default value is `1`.
    apod_factor : bool, optional
        Apply apodization factor or not. Default value is `False`.
    envelope : float, optional
        Size :math:`k_{\mathrm{env}}` of the Gaussian envelope :math:`A(\mathbf{s}) = \mathrm{e}^{-(k^2_x+k^2_y)/k_\mathrm{env}^2}`.
        Default is `None`.
    gibson_lanni : bool, optional
        Apply Gibson-Lanni aberration correction or not. Default value is `False`.
    z_p : float, optional
        Depth of the focal plane in the sample. It is usually obtained experimentally by focusing on a point source
        at this depth.  Default value is `1e3`.
    n_s : float, optional
        Refractive index of the sample. Default value is `1.3`.
    n_g : float, optional
        Refractive index of the (glass) cover slip. Default value is `1.5`.
    n_g0 : float, optional
        Design condition of the refractive index of the cover slip. Default value is `1.5`.
    t_g : float, optional
        Thickness of the (glass) cover slip. Default value is `170e3`.
    t_g0 : float, optional
        Design condition of the thickness of the (glass) cover slip. Default value is `170e3`.
    n_i : float, optional
        Refractive index of the immersion medium. Default value is `1.5`.
    n_i0 : float, optional
        Design condition of the refractive index of the immersion medium. Default value is `1.5`.
    t_i0 : float, optional
        Design condition of the thickness of the immersion medium. Default value is `100e3`.

    Notes
    -----
    Internal parameters:

    1. t_i : float,
    thickness of the immersion medium for which an emitter at depth :math:`z_p` is paraxially in focus. It is
    computed from
    :math:`t_i = n_i \left( \frac{t_g^0}{n_g^0} + \frac{t_i^0}{n_i^0} - \frac{t_g}{n_g} - \frac{z_p}{n_s} \right)`,
    i.e. Eq. (3.56) of Aguet's thesis (https://bigwww.epfl.ch/publications/aguet0903.pdf) with zero defocus:
    the defocus :math:`z` of the library is applied separately by the propagation kernel, not by moving the
    objective.

    2. refractive_index : float,
    refractive index of the propagation medium. It is equal to :math:`n_s` if gibson_lanni=True, :math:`n_i`, otherwise.

    3. `(z_p, n_s, n_g, n_g0, t_g, t_g0, n_i, t_i0, t_i)` are coefficients related to the aberrations due to refractive
    index mismatch between stratified layers of the microscope.
    This aberration is computed by method `self.compute_optical_path`.

    4. x : torch.Tensor of shape `(n_pix_psf,)`,
    physical lateral coordinates of the PSF grid in nanometer (identical along both lateral axes):
    ``x[i] = (i - n_pix_psf // 2) * pix_size``.

    5. z : torch.Tensor of shape `(n_defocus,)`,
    physical axial coordinates of the z-slices in nanometer: ``z[i] = (i - n_defocus // 2) * defocus_step``.

    """

    #: Key under which the name of the propagator is stored by :meth:`to_dict`.
    _registry_key = 'propagator'

    @classmethod
    def _registry(cls) -> dict:
        from . import PROPAGATORS
        return PROPAGATORS

    def __init__(self,
                 n_pix_pupil: int =128,
                 n_pix_psf: int = 128,
                 device: str = 'cpu',
                 zernike_coefficients=None,
                 wavelength: float = 632,
                 na: float = 1.3,
                 pix_size: float = 20,
                 defocus_step: float = 0.0,
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
        _validate_device(device)
        # Both parameterisations sample the pupil with a step of 1 / (n_pix_pupil - 1).
        _validate_number('n_pix_pupil', n_pix_pupil, 2)
        _validate_number('n_pix_psf', n_pix_psf, 1)
        _validate_number('n_defocus', n_defocus, 1)
        _validate_number('wavelength', wavelength, 0, strict=True)
        _validate_number('pix_size', pix_size, 0, strict=True)
        _validate_number('na', na, 0, strict=True)
        # Beyond na = n_i0 the pupil is not physical: sin(theta_max) > 1, which makes the spherical propagator
        # return NaN and the Cartesian one a meaningless field.
        _validate_number('n_i0', n_i0, 0, strict=True)
        if na > n_i0:
            raise ValueError(f'The numerical aperture cannot exceed the design refractive index of the immersion '
                             f'medium: got na={na!r} and n_i0={n_i0!r}.')
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
        self.pix_size = pix_size
        self.fov = pix_size * n_pix_psf
        self.defocus_step = defocus_step
        self.n_defocus = n_defocus
        # Physical coordinates of the PSF grid. Every axis is centred on index n // 2, so the optical
        # axis (x = y = 0) and the focal plane (z = 0) are sampled exactly, and neighbouring samples are
        # exactly pix_size (resp. defocus_step) apart.
        self.x = _centred_grid(n_pix_psf, pix_size)
        self.z = _centred_grid(n_defocus, defocus_step)
        self.defocus_min = float(self.z[0])
        self.defocus_max = float(self.z[-1])
        # Zernike basis (cached per number of coefficients) and the resulting phase aberration; both are
        # computed by the subclasses once the pupil geometry is known.
        self._zernike_basis = None
        self._zernike_aberrations = None
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
            self.refractive_index = n_i

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Get name of the propagator in a certain format, e.g. 'scalar_cartesian'."""
        raise NotImplementedError

    @abstractmethod
    def initialize_input_field(self) -> torch.Tensor:
        """Initialize the input field of propagator."""
        raise NotImplementedError

    @abstractmethod
    def get_pupil(self) -> torch.Tensor:
        """Get the pupil function with all corrections applied."""
        raise NotImplementedError

    @abstractmethod
    def compute_focus_field(self) -> torch.Tensor:
        """Compute the output field of the propagator at focal plane."""
        raise NotImplementedError

    #: Mesh on which the Zernike polynomials are evaluated ('cartesian' or 'spherical'), set by the subclasses.
    _zernike_mesh_type: str = 'cartesian'

    def _zernike_radius(self):
        """Normalized radius of every pupil sample, or None if the mesh of :func:`zernike_basis` already knows it.

        Only used by the spherical mesh, whose samples are equispaced in the polar angle rather than in the
        radius; see :meth:`SphericalPropagator._zernike_radius`.
        """
        return None

    def update_zernike_coefficients(self, zernike_coefficients):
        """Update Zernike coefficients without reinitializing propagator."""
        if not isinstance(zernike_coefficients, torch.Tensor):
            zernike_coefficients = torch.tensor(zernike_coefficients)
        self.zernike_coefficients = zernike_coefficients
        self._compute_zernike_aberrations()

    def _compute_zernike_aberrations(self):
        """(Re)compute the Zernike phase aberration of the pupil from ``self.zernike_coefficients``.

        The Zernike basis is built once for the current number of coefficients and cached on the device,
        so that updating the coefficients (e.g. inside an optimization loop) only costs a weighted sum.
        """
        n_modes = len(self.zernike_coefficients)
        if self._zernike_basis is None or self._zernike_basis.shape[0] != n_modes:
            self._zernike_basis = zernike_basis(n_modes, self.n_pix_pupil, self._zernike_mesh_type,
                                                rho=self._zernike_radius()).to(self.device)
        self._zernike_aberrations = create_zernike_aberrations(
            self.zernike_coefficients, self.n_pix_pupil, self._zernike_mesh_type, basis=self._zernike_basis)

    def compute_optical_path(self, sin_t: torch.Tensor) -> torch.Tensor:
        r"""Compute the optical path following Eq. (3.45) in [1]_.

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
        """Constructor arguments of the propagator as JSON-serialisable values (see :meth:`to_dict`)."""
        return {
            'n_pix_pupil': self.n_pix_pupil,
            'n_pix_psf': self.n_pix_psf,
            'device': str(self.device),
            'zernike_coefficients': convert_tensor_to_array(self.zernike_coefficients).tolist(),
            'wavelength': self.wavelength,
            'na': self.na,
            'pix_size': self.pix_size,
            'defocus_step': self.defocus_step,
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
            'n_i0': self.n_i0,
            't_i0': self.t_i0,
        }

    @classmethod
    def _decode_args(cls, args: dict) -> dict:
        """
        Inverse of :meth:`_get_args`: turn the JSON values back into constructor arguments.

        Files written by versions <= 0.1.0 are accepted: the derived values they contain are dropped.
        """
        args = dict(args)
        for key in _DERIVED_KEYS:
            args.pop(key, None)
        return args
