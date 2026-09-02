# Copyright Biomedical Imaging Group, EPFL 2025

r"""
The abstract dipole imager: the image of a radiating dipole formed by an aplanatic microscope.

The propagators of :mod:`psf_generator.propagators` describe the *focusing* (illumination) path of a microscope:
a pupil function is turned into a focus field. The dipole imagers describe the reverse, *imaging* (detection)
path: a dipole radiating in the sample -- a fluorophore, or the dipole induced in a nanoparticle by the
illumination -- is collected by the objective and imaged onto a camera by the tube lens. Both paths are governed
by the Debye-Wolf integral, but running it backwards changes a few geometric factors (apodization, Jacobian,
Fresnel coefficients, radiation pattern), see the theory section of the documentation.

"""
import math
import numbers
import typing as tp
import warnings
from abc import ABC, abstractmethod

import torch

from ..utils.misc import centred_grid
from ..utils.parameters import Parametrized, validate_device, validate_number
from ..utils.zernike import create_zernike_aberrations, zernike_basis

#: Ways of modelling the Fresnel transmission of the light radiated by the dipole through the interfaces of the
#: sample, see :class:`DipoleImager`.
FRESNEL_MODES = ('reciprocal', 'forward', 'none')


def _transmission_s(n_1, cos_1, n_2, cos_2):
    """Fresnel amplitude transmission coefficient of s-polarized light going from medium 1 into medium 2."""
    return 2 * n_1 * cos_1 / (n_1 * cos_1 + n_2 * cos_2)


def _transmission_p(n_1, cos_1, n_2, cos_2):
    """Fresnel amplitude transmission coefficient of p-polarized light going from medium 1 into medium 2."""
    return 2 * n_1 * cos_1 / (n_2 * cos_1 + n_1 * cos_2)


class DipoleImager(Parametrized, ABC):
    r"""
    Base class of the dipole imagers, which compute the image of a radiating dipole (the detection path).

    A dipole located at :math:`\mathbf{r}_p = (x_p, y_p, z_p)` in the sample medium radiates the far field
    :math:`\mathbf{E}(\mathbf{r}) = [(\hat{\mathbf{s}} \times \mathbf{p}) \times \hat{\mathbf{s}}] \,
    \mathrm{e}^{\mathrm{i} k_s r} / r` (:math:`k_s = 2\pi n_s / \lambda`). The light crosses the coverslip and the
    immersion medium, is collimated by an aplanatic objective of numerical aperture NA and focused onto the
    camera by a low-NA tube lens. The image field, expressed in *object-space coordinates* (the camera
    coordinates divided by the magnification; a common factor :math:`\sqrt{n_i / n_d} / M` is dropped), is

    .. math::

        \hat{\mathbf{E}}(\boldsymbol{\rho}) = -\frac{\mathrm{i} k_i}{2\pi}
        \iint\limits_{s_x^2 + s_y^2 \leq s_{\max}^2}
        \frac{\tilde{\mathbf{e}}(\mathbf{s})}{\sqrt{\cos\theta}} \,
        \mathrm{e}^{\mathrm{i} k_i \mathbf{s} \cdot \boldsymbol{\rho}} \, \mathrm{d}s_x \, \mathrm{d}s_y
        = -\frac{\mathrm{i} k_i}{2\pi} \int_0^{\theta_{\max}} \mathrm{d}\theta \int_0^{2\pi} \mathrm{d}\phi \,
        \tilde{\mathbf{e}}(\theta, \phi) \sqrt{\cos\theta} \sin\theta \,
        \mathrm{e}^{\mathrm{i} k_i \mathbf{s} \cdot \boldsymbol{\rho}},

    with :math:`k_i = 2\pi n_i / \lambda`, :math:`\theta` the polar angle in the immersion medium,
    :math:`s_{\max} = \sin\theta_{\max} = \mathrm{NA} / n_i^0`, and :math:`\tilde{\mathbf{e}}` the far field
    of the dipole *in the immersion medium*, rotated into the pupil plane by the objective. Compared to the
    focusing integral of the propagators, the imaging integral

    - uses the apodization :math:`1/\sqrt{\cos\theta}` (sphere to plane) instead of :math:`\sqrt{\cos\theta}`,
    - has no :math:`1/s_z` Jacobian (the tube lens has a low NA), so that in spherical coordinates the two
      integrals have the same weight :math:`\sqrt{\cos\theta} \sin\theta`,
    - multiplies the radiation pattern of the dipole, evaluated at the angle :math:`\theta_s` *in the sample*
      (:math:`n_s \sin\theta_s = n_i \sin\theta`), by the Fresnel coefficients of the interfaces. The far field
      of a dipole seen from a medium of higher index is :math:`t_{s \to i} \, n_i \cos\theta / (n_s \cos\theta_s)`
      times its far field in the sample, which is exactly the Fresnel transmission coefficient of the *reverse*
      direction, :math:`t_{i \to s}` (Novotny & Hecht, Eqs. 10.36-10.38; Foreman & Török 2011). This is the
      default ``fresnel='reciprocal'``; ``'forward'`` uses :math:`t_{s \to i}` without the geometric factor,
      as in Mahmoodabadi et al. 2020 and Dong et al. 2021, and ``'none'`` sets the coefficients to one.
    - accumulates the phase :math:`k \Lambda(\theta)` of the optical path from the dipole to the objective
      through the sample (thickness :math:`z_p`), the coverslip (:math:`t_g`, :math:`n_g`) and the immersion
      medium (:math:`t_i`, :math:`n_i`), relative to the design conditions (starred quantities):

      .. math::

          \Lambda(\theta) = z_p \sqrt{n_s^2 - n_i^2 \sin^2\theta} + t_i \, n_i \cos\theta
          + t_g \sqrt{n_g^2 - n_i^2 \sin^2\theta} - t_g^* \sqrt{{n_g^*}^2 - n_i^2 \sin^2\theta}
          - t_i^* \sqrt{{n_i^*}^2 - n_i^2 \sin^2\theta}.

      Beyond the critical angle of the sample (:math:`n_i \sin\theta > n_s`) the square root is imaginary: these
      are the evanescent components of the dipole field, which are collected when the dipole is close to the
      coverslip (supercritical angle emission) and decay with :math:`z_p`.

    The thickness of the immersion medium follows from the axial position of the focus: a dipole at
    :math:`z_p = z_{\mathrm{focus}}` is in (paraxial) focus, i.e.
    :math:`t_i = n_i (t_g^* / n_g^* + t_i^* / n_i^* - t_g / n_g - z_{\mathrm{focus}} / n_s)`, the same
    convention as the parameter ``z_p`` of the propagators. Aberrations of the detection path can be added as
    Zernike modes on the pupil, like for the propagators.

    Parameters
    ----------
    n_pix_pupil : int, optional
        Number of samples of the pupil (per axis for the Cartesian imager, along the polar angle for the
        spherical one). Default value is `128`.
    n_pix_psf : int, optional
        Number of pixels (size) of the image (always square). Default value is `128`.
    device : str or torch.device, optional
        Computational backend, e.g. `'cpu'`, `'cuda'` or `'mps'`. Default value is `'cpu'`.
    zernike_coefficients : np.ndarray or torch.tensor, optional
        Zernike coefficients (OSA order, radians) of the aberrations of the detection path. Default is `None`.
    wavelength : float, optional
        Wavelength of light, in nanometer. Default value is `632`.
    na : float, optional
        Numerical aperture of the objective. Default value is `1.3`.
    pix_size : float, optional
        Pixel size of the image in object space (camera pixel size divided by the magnification), in
        nanometer. Pixel ``i`` is located at :math:`x_i = (i - \lfloor n_{\mathrm{pix}}/2 \rfloor)\, \mathrm{pix\_size}`
        (see attribute ``x``). Default value is `20`.
    z_focus : float, optional
        Axial position of the focal plane in the sample, in nanometer, measured from the coverslip-sample
        interface towards the sample (a dipole at ``z_p = z_focus`` is in paraxial focus). Default value is `0`.
    n_s : float, optional
        Refractive index of the sample medium. Default value is `1.33`.
    n_g : float, optional
        Refractive index of the (glass) cover slip. Default value is `1.5`.
    n_g0 : float, optional
        Design condition of the refractive index of the cover slip. Default value is `1.5`.
    t_g : float, optional
        Thickness of the cover slip, in nanometer. Default value is `170e3`.
    t_g0 : float, optional
        Design condition of the thickness of the cover slip. Default value is `170e3`.
    n_i : float, optional
        Refractive index of the immersion medium. Default value is `1.5`.
    n_i0 : float, optional
        Design condition of the refractive index of the immersion medium. Default value is `1.5`.
    t_i0 : float, optional
        Design condition of the thickness of the immersion medium, in nanometer. Default value is `100e3`.
    fresnel : str, optional
        `'reciprocal'` (physical, default), `'forward'` or `'none'`, see above.

    Notes
    -----
    Internal parameters:

    1. t_i : float, thickness of the immersion medium, computed from `z_focus` (see above).

    2. x : torch.Tensor of shape `(n_pix_psf,)`, physical lateral coordinates of the image grid in nanometer
    (identical along both axes). The image tensors are indexed as ``[..., y, x]``.

    3. The sign conventions: the :math:`z` axis points from the objective into the sample, so that
    :math:`z_p > 0` is a dipole above the coverslip, farther away from the objective. A dipole displaced by
    :math:`\Delta z` produces the same image as the focus field of the propagators at defocus :math:`\Delta z`.

    """

    #: Key under which the name of the imager is stored by :meth:`to_dict`.
    _registry_key = 'imager'

    #: Mesh on which the Zernike polynomials are evaluated ('cartesian' or 'spherical'), set by the subclasses.
    _zernike_mesh_type: str = 'cartesian'

    def __init__(self,
                 n_pix_pupil: int = 128,
                 n_pix_psf: int = 128,
                 device: str = 'cpu',
                 zernike_coefficients=None,
                 wavelength: float = 632,
                 na: float = 1.3,
                 pix_size: float = 20,
                 z_focus: float = 0.0,
                 n_s: float = 1.33,
                 n_g: float = 1.5,
                 n_g0: float = 1.5,
                 t_g: float = 170e3,
                 t_g0: float = 170e3,
                 n_i: float = 1.5,
                 n_i0: float = 1.5,
                 t_i0: float = 100e3,
                 fresnel: str = 'reciprocal'):
        validate_device(device)
        validate_number('n_pix_pupil', n_pix_pupil, 2)
        validate_number('n_pix_psf', n_pix_psf, 1)
        validate_number('wavelength', wavelength, 0, strict=True)
        validate_number('pix_size', pix_size, 0, strict=True)
        validate_number('na', na, 0, strict=True)
        for name, value in (('n_s', n_s), ('n_g', n_g), ('n_g0', n_g0), ('n_i', n_i), ('n_i0', n_i0)):
            validate_number(name, value, 0, strict=True)
        for name, value in (('t_g', t_g), ('t_g0', t_g0), ('t_i0', t_i0)):
            validate_number(name, value, 0)
        if not isinstance(z_focus, numbers.Real) or isinstance(z_focus, bool):
            raise ValueError(f'z_focus must be a number, got {z_focus!r}.')
        if na > n_i0:
            raise ValueError(f'The numerical aperture cannot exceed the design refractive index of the immersion '
                             f'medium: got na={na!r} and n_i0={n_i0!r}.')
        if fresnel not in FRESNEL_MODES:
            raise ValueError(f'Unknown fresnel mode {fresnel!r}, choose from {FRESNEL_MODES}.')
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
        self.z_focus = z_focus
        self.n_s = n_s
        self.n_g = n_g
        self.n_g0 = n_g0
        self.t_g = t_g
        self.t_g0 = t_g0
        self.n_i = n_i
        self.n_i0 = n_i0
        self.t_i0 = t_i0
        self.fresnel = fresnel
        # Vacuum wavenumber (1/nm) and pupil cut-off.
        self.k = 2.0 * math.pi / wavelength
        self.s_max = na / n_i0
        # Image grid (object space, pixel-centred, see ``Propagator.x``).
        self.x = centred_grid(n_pix_psf, pix_size)
        # Thickness of the immersion medium: paraxial focus at depth z_focus in the sample.
        self.t_i = n_i * (t_g0 / n_g0 + t_i0 / n_i0 - t_g / n_g - z_focus / n_s)
        self._zernike_basis = None
        self._zernike_aberrations = None

    @classmethod
    def _registry(cls) -> dict:
        from . import IMAGERS
        return IMAGERS

    # ------------------------------------------------------------------------------------------------------------
    # Layered medium: angles, Fresnel coefficients, optical path
    # ------------------------------------------------------------------------------------------------------------
    def _layer_factors(self, sin_t: torch.Tensor) -> tp.Dict[str, torch.Tensor]:
        r"""
        Angular factors of the imaging integral for the given sines of the polar angle in the immersion medium.

        Parameters
        ----------
        sin_t : torch.Tensor
            :math:`\sin\theta` of every pupil sample (any shape, :math:`\leq 1`).

        Returns
        -------
        factors : dict of torch.Tensor (float64 / complex128, same shape as `sin_t`)
            ``cos_t`` (:math:`\cos\theta`, real), ``sin_s`` and ``cos_s`` (angle in the sample, complex beyond
            the critical angle), ``t_s`` and ``t_p`` (Fresnel amplitude coefficients of the s and p components
            through both interfaces), ``path0`` (optical path :math:`\Lambda` in nm without the term in
            :math:`z_p`) and ``path_z`` (its coefficient :math:`\sqrt{n_s^2 - n_i^2 \sin^2\theta}`).

        """
        sin_t = sin_t.to(torch.float64)
        sin_sq = sin_t ** 2
        cos_t = torch.sqrt((1.0 - sin_sq).clamp(min=0.0))

        def cosine_in(n: float) -> torch.Tensor:
            # Cosine of the ray angle in a medium of index n with the same transverse wavenumber; the principal
            # square root gives +i * |...| beyond the critical angle, i.e. a wave decaying away from the interface.
            return torch.sqrt((1.0 - (self.n_i / n) ** 2 * sin_sq).to(torch.complex128))

        cos_g, cos_s = cosine_in(self.n_g), cosine_in(self.n_s)
        cos_g0, cos_i0 = cosine_in(self.n_g0), cosine_in(self.n_i0)
        cos_i = cos_t.to(torch.complex128)
        if self.fresnel == 'reciprocal':
            # Coefficients of the reverse direction (immersion -> glass -> sample): they equal the forward ones
            # times the geometric factor n_i cos(theta) / (n_s cos(theta_s)) of the far field across interfaces.
            t_s = _transmission_s(self.n_i, cos_i, self.n_g, cos_g) * _transmission_s(self.n_g, cos_g, self.n_s, cos_s)
            t_p = _transmission_p(self.n_i, cos_i, self.n_g, cos_g) * _transmission_p(self.n_g, cos_g, self.n_s, cos_s)
        elif self.fresnel == 'forward':
            t_s = _transmission_s(self.n_s, cos_s, self.n_g, cos_g) * _transmission_s(self.n_g, cos_g, self.n_i, cos_i)
            t_p = _transmission_p(self.n_s, cos_s, self.n_g, cos_g) * _transmission_p(self.n_g, cos_g, self.n_i, cos_i)
        else:
            t_s = torch.ones_like(cos_s)
            t_p = torch.ones_like(cos_s)
        # At grazing incidence (cos = 0 on both sides of an index-matched interface) the coefficients are 0/0;
        # their limit is zero.
        t_s = torch.nan_to_num(t_s, nan=0.0)
        t_p = torch.nan_to_num(t_p, nan=0.0)
        path0 = (self.t_i * self.n_i * cos_i + self.t_g * self.n_g * cos_g
                 - self.t_g0 * self.n_g0 * cos_g0 - self.t_i0 * self.n_i0 * cos_i0)
        path_z = self.n_s * cos_s
        sin_s = (self.n_i / self.n_s * sin_t).to(torch.complex128)
        return {'cos_t': cos_t, 'sin_s': sin_s, 'cos_s': cos_s, 't_s': t_s, 't_p': t_p,
                'path0': path0, 'path_z': path_z}

    @staticmethod
    def _pattern_factors(factors: tp.Dict[str, torch.Tensor]) -> tp.Tuple[torch.Tensor, ...]:
        r"""
        Angular weights :math:`A_0 = t_p \cos\theta_s + t_s`, :math:`A_1 = t_p \sin\theta_s` and
        :math:`A_2 = t_p \cos\theta_s - t_s` of the far field of the dipole in the pupil plane.

        The pupil field reads :math:`\tilde{e}_x = p_x (A_0 + A_2 \cos 2\phi)/2 + p_y A_2 \sin 2\phi / 2
        + p_z A_1 \cos\phi` and :math:`\tilde{e}_y = p_x A_2 \sin 2\phi / 2 + p_y (A_0 - A_2 \cos 2\phi)/2
        + p_z A_1 \sin\phi`; in a homogeneous medium they reduce to the :math:`q_i` of Foreman & Török 2011.
        """
        a_0 = factors['t_p'] * factors['cos_s'] + factors['t_s']
        a_2 = factors['t_p'] * factors['cos_s'] - factors['t_s']
        a_1 = factors['t_p'] * factors['sin_s']
        return a_0, a_1, a_2

    # ------------------------------------------------------------------------------------------------------------
    # Zernike aberrations of the detection path
    # ------------------------------------------------------------------------------------------------------------
    def _zernike_radius(self):
        """Normalized radius of every pupil sample (spherical mesh only), see ``SphericalPropagator``."""
        return None

    def update_zernike_coefficients(self, zernike_coefficients) -> None:
        """Update Zernike coefficients without reinitializing the imager."""
        if not isinstance(zernike_coefficients, torch.Tensor):
            zernike_coefficients = torch.tensor(zernike_coefficients)
        self.zernike_coefficients = zernike_coefficients
        self._compute_zernike_aberrations()

    def _compute_zernike_aberrations(self) -> None:
        """(Re)compute the Zernike phase aberration of the pupil from ``self.zernike_coefficients``."""
        n_modes = len(self.zernike_coefficients)
        if self._zernike_basis is None or self._zernike_basis.shape[0] != n_modes:
            self._zernike_basis = zernike_basis(n_modes, self.n_pix_pupil, self._zernike_mesh_type,
                                                rho=self._zernike_radius()).to(self.device)
        self._zernike_aberrations = create_zernike_aberrations(
            self.zernike_coefficients, self.n_pix_pupil, self._zernike_mesh_type, basis=self._zernike_basis)

    # ------------------------------------------------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _as_dipole(dipole) -> torch.Tensor:
        """Dipole moment as a complex128 tensor of shape (3,) on the CPU."""
        dipole = torch.as_tensor(dipole).detach().cpu().reshape(-1).to(torch.complex128)
        if dipole.shape != (3,):
            raise ValueError(f'The dipole must have three components (p_x, p_y, p_z), got {tuple(dipole.shape)}.')
        return dipole

    @staticmethod
    def _as_positions(positions) -> torch.Tensor:
        """Positions as a float64 tensor of shape (n_positions, 3) on the CPU."""
        positions = torch.as_tensor(positions).detach().cpu().to(torch.float64)
        if positions.numel() == 0 or positions.numel() % 3 != 0 or positions.shape[-1] != 3:
            raise ValueError(f'positions must be an array of shape (3,) or (n_positions, 3) holding (x_p, y_p, z_p) '
                             f'in nanometer, got shape {tuple(positions.shape)}.')
        return positions.reshape(-1, 3)

    # ------------------------------------------------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def _compute_image_xy(self, dipole: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Transverse components of the image field, of shape `(n_positions, 2, n_pix_psf, n_pix_psf)`."""
        raise NotImplementedError

    def compute_image(self, dipole=(1.0, 0.0, 0.0), positions=(0.0, 0.0, 0.0)) -> torch.Tensor:
        r"""
        Compute the image field of a dipole for one or several positions.

        Parameters
        ----------
        dipole : array-like of 3 complex numbers, optional
            Far-field amplitude vector :math:`\mathbf{p} = (p_x, p_y, p_z)` of the dipole in the sample medium,
            such that it radiates :math:`\mathbf{E}(\mathbf{r}) = [(\hat{\mathbf{s}} \times \mathbf{p}) \times
            \hat{\mathbf{s}}] \, \mathrm{e}^{\mathrm{i} k_s r} / r`. For a fluorophore this is its orientation (in
            arbitrary units); for a Rayleigh scatterer of polarizability :math:`\alpha` (in nm\ :sup:`3`) in the
            field :math:`\mathbf{E}_{\mathrm{inc}}`, :math:`\mathbf{p} = k_s^2 \alpha \mathbf{E}_{\mathrm{inc}} / (4\pi)`.
            Default is an x-oriented unit dipole.
        positions : array-like of shape `(3,)` or `(n_positions, 3)`, optional
            Positions :math:`(x_p, y_p, z_p)` of the dipole in nanometer: lateral position in the image grid and
            height above the coverslip. Default is the origin.

        Returns
        -------
        field : torch.Tensor
            Complex image field of shape `(n_positions, 3, n_pix_psf, n_pix_psf)` in object-space units (see the
            class documentation). The third component (:math:`E_z`) is zero: the tube lens has a low NA.

        """
        dipole = self._as_dipole(dipole)
        positions = self._as_positions(positions)
        field_xy = self._compute_image_xy(dipole, positions)
        field_z = torch.zeros_like(field_xy[:, :1])
        return torch.cat([field_xy, field_z], dim=1)

    # ------------------------------------------------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------------------------------------------------
    def _get_args(self) -> dict:
        return {
            'n_pix_pupil': self.n_pix_pupil,
            'n_pix_psf': self.n_pix_psf,
            'device': str(self.device),
            'zernike_coefficients': self.zernike_coefficients.detach().cpu().tolist(),
            'wavelength': self.wavelength,
            'na': self.na,
            'pix_size': self.pix_size,
            'z_focus': self.z_focus,
            'n_s': self.n_s,
            'n_g': self.n_g,
            'n_g0': self.n_g0,
            't_g': self.t_g,
            't_g0': self.t_g0,
            'n_i': self.n_i,
            'n_i0': self.n_i0,
            't_i0': self.t_i0,
            'fresnel': self.fresnel,
        }

    def _warn_custom_field_not_saved(self) -> None:
        if getattr(self, 'custom_field', None) is not None:
            warnings.warn('The custom_field tensor cannot be saved to JSON and is not stored.', stacklevel=4)
