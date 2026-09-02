# Copyright Biomedical Imaging Group, EPFL 2025

r"""
Zernike polynomials and special phase masks for the pupil function.

Zernike polynomials are identified by the OSA/ANSI single index

.. math:: j = \frac{n (n + 2) + l}{2},

where :math:`n \geq 0` is the radial order and :math:`l \in \{-n, -n + 2, \ldots, n\}` the azimuthal frequency
(:math:`l < 0` for the :math:`\sin` modes and :math:`l \geq 0` for the :math:`\cos` modes). The first modes are

===  ===  ===  ==============================
  j    n    l  name
===  ===  ===  ==============================
  0    0    0  piston
  1    1   -1  vertical tilt
  2    1    1  horizontal tilt
  3    2   -2  oblique astigmatism
  4    2    0  defocus
  5    2    2  vertical astigmatism
  6    3   -3  vertical trefoil
  7    3   -1  vertical coma
  8    3    1  horizontal coma
  9    3    3  oblique trefoil
 10    4   -4  oblique quadrafoil
 11    4   -2  oblique secondary astigmatism
 12    4    0  primary spherical
===  ===  ===  ==============================

The polynomials are evaluated on the unit disk :math:`\rho \leq 1` (zero outside) and are **not** normalized:
every mode has unit peak amplitude, so a Zernike coefficient is the peak phase of that mode in radians.

"""
import math
import typing as tp
import warnings

import torch

__all__ = [
    'create_pupil_mesh',
    'osa_index_to_nl',
    'nl_to_osa_index',
    'zernike_polynomial',
    'zernike_basis',
    'create_zernike_aberrations',
    'create_special_pupil',
]


def create_pupil_mesh(n_pixels: int) -> tuple[torch.Tensor, ...]:
    """
    Create a 2D square meshgrid for the pupil function.

    Parameters
    ----------
    n_pixels : int
        Number of pixels for the pupil function.

    Returns
    -------
    (kx, ky): Tuple[torch.Tensor, ...]
        Two Tensors that represent the 2D coordinates on the mesh.

    """
    x = torch.linspace(-1, 1, n_pixels)
    y = torch.linspace(-1, 1, n_pixels)
    kx, ky = torch.meshgrid(x, y, indexing='xy')
    return kx, ky


def osa_index_to_nl(index: int) -> tuple[int, int]:
    r"""
    Convert the OSA/ANSI single index :math:`j` of a Zernike polynomial to its :math:`(n, l)` pair.

    The index is defined as :math:`j = (n(n + 2) + l) / 2`; the radial order :math:`n` is the largest integer
    with :math:`n(n + 1) / 2 \leq j`, and :math:`l = 2j - n(n + 2)`.

    Parameters
    ----------
    index : int
        OSA index :math:`j \geq 0`.

    Returns
    -------
    (n, l) : Tuple[int, int]
        Radial order and azimuthal frequency.

    """
    if index < 0:
        raise ValueError(f'The OSA index must be a non-negative integer, not {index}.')
    n = (math.isqrt(8 * index + 1) - 1) // 2
    l = 2 * index - n * (n + 2)
    return n, l


def nl_to_osa_index(n: int, l: int) -> int:
    r"""
    Convert the :math:`(n, l)` pair of a Zernike polynomial to its OSA/ANSI single index :math:`j = (n(n + 2) + l) / 2`.

    Parameters
    ----------
    n : int
        Radial order, :math:`n \geq 0`.
    l : int
        Azimuthal frequency, :math:`|l| \leq n` and :math:`n - l` even.

    Returns
    -------
    index : int
        OSA index.

    """
    if n < 0 or abs(l) > n or (n - l) % 2:
        raise ValueError(f'Invalid Zernike indices (n, l) = ({n}, {l}): need n >= 0, |l| <= n and n - l even.')
    return (n * (n + 2) + l) // 2


def zernike_polynomial(n: int, l: int, rho: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    r"""
    Evaluate the (unnormalized) Zernike polynomial :math:`Z_n^l(\rho, \phi)`.

    .. math::

        Z_n^l(\rho, \phi) = R_n^{|l|}(\rho) \times \begin{cases} \cos(|l| \phi) & l \geq 0 \\ \sin(|l| \phi) & l < 0 \end{cases},
        \qquad
        R_n^m(\rho) = \sum_{k=0}^{(n - m)/2} (-1)^k \binom{n - k}{k} \binom{n - 2k}{(n - m)/2 - k} \rho^{n - 2k},

    and :math:`Z_n^l = 0` outside the unit disk :math:`\rho > 1`.

    Parameters
    ----------
    n : int
        Radial order, :math:`n \geq 0`.
    l : int
        Azimuthal frequency, :math:`|l| \leq n` and :math:`n - l` even.
    rho : torch.Tensor
        Radial coordinate, :math:`\rho \geq 0`.
    phi : torch.Tensor
        Azimuthal angle, same shape as `rho`.

    Returns
    -------
    Z : torch.Tensor
        :math:`Z_n^l` evaluated at `(rho, phi)`, same shape and dtype as `rho`.

    """
    m = abs(l)
    if n < 0 or m > n or (n - m) % 2:
        raise ValueError(f'Invalid Zernike indices (n, l) = ({n}, {l}): need n >= 0, |l| <= n and n - l even.')
    radial = torch.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coefficient = (-1) ** k * math.comb(n - k, k) * math.comb(n - 2 * k, (n - m) // 2 - k)
        radial = radial + coefficient * rho ** (n - 2 * k)
    angular = torch.cos(m * phi) if l >= 0 else torch.sin(m * phi)
    return torch.where(rho <= 1, radial * angular, torch.zeros_like(radial))


def zernike_basis(n_modes: int, n_pix_pupil: int, mesh_type: str = 'cartesian',
                  rho: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Stack the first `n_modes` Zernike polynomials (OSA order) sampled on the pupil.

    Parameters
    ----------
    n_modes : int
        Number of modes, i.e. OSA indices :math:`0, \ldots, n_{\mathrm{modes}} - 1`.
    n_pix_pupil : int
        Number of pixels of the pupil function.
    mesh_type : str
        'cartesian': modes are sampled on the square grid of :func:`create_pupil_mesh`, giving a tensor of
        shape `(n_modes, n_pix_pupil, n_pix_pupil)`.
        'spherical': modes are sampled along the radius only, giving a tensor of shape `(n_modes, n_pix_pupil)`.
        The spherical propagators assume an axisymmetric pupil, so the modes with :math:`l \neq 0` are
        identically zero on this mesh.
    rho : torch.Tensor, optional
        Normalized radius :math:`\rho \in [0, 1]` of every pupil sample, of shape `(n_pix_pupil,)`. Only for
        the spherical mesh; if None, the samples are assumed to be equispaced in :math:`\rho`
        (``torch.linspace(0, 1, n_pix_pupil)``). The spherical propagators sample the pupil uniformly in the
        polar angle :math:`\theta` instead, so they pass their own radii
        :math:`\rho_i = \sin\theta_i / \sin\theta_{\max}`.

    Returns
    -------
    basis : torch.Tensor
        Zernike modes of dtype `torch.float32`.

    """
    if n_modes < 1:
        raise ValueError(f'At least one Zernike mode is required, got n_modes={n_modes}.')
    if mesh_type == 'cartesian':
        if rho is not None:
            raise ValueError('A custom radius rho is only supported by the spherical mesh; the Cartesian mesh '
                             'samples the pupil on the square grid of create_pupil_mesh.')
        kx, ky = create_pupil_mesh(n_pixels=n_pix_pupil)
        kx, ky = kx.to(torch.float64), ky.to(torch.float64)
        rho = torch.sqrt(kx ** 2 + ky ** 2)
        phi = torch.atan2(ky, kx)
    elif mesh_type == 'spherical':
        if rho is None:
            rho = torch.linspace(0, 1, n_pix_pupil, dtype=torch.float64)
        else:
            if not isinstance(rho, torch.Tensor):
                rho = torch.as_tensor(rho)
            if rho.shape != (n_pix_pupil,):
                raise ValueError(f'The radius rho must be a 1D tensor of shape ({n_pix_pupil},), '
                                 f'got {tuple(rho.shape)}.')
            rho = rho.detach().cpu().to(torch.float64)
        phi = torch.zeros_like(rho)
    else:
        raise ValueError(f"Invalid mesh type {mesh_type}, choose 'spherical' or 'cartesian'.")

    modes = []
    for index in range(n_modes):
        n, l = osa_index_to_nl(index)
        if mesh_type == 'spherical' and l != 0:
            modes.append(torch.zeros_like(rho))
        else:
            modes.append(zernike_polynomial(n, l, rho, phi))
    return torch.stack(modes).to(torch.float32)


def _warn_if_not_axisymmetric(zernike_coefficients: torch.Tensor) -> None:
    """Warn about non-zero coefficients of modes that the spherical mesh cannot represent."""
    ignored = [index for index, coefficient in enumerate(zernike_coefficients.detach().cpu().tolist())
               if coefficient != 0 and osa_index_to_nl(index)[1] != 0]
    if ignored:
        warnings.warn(f'Zernike modes {ignored} are not axisymmetric and are ignored in spherical coordinates; '
                      f'use a Cartesian propagator to apply them.', stacklevel=3)


def create_zernike_aberrations(
        zernike_coefficients: tp.Union[torch.Tensor, tp.Sequence[float]],
        n_pix_pupil: int,
        mesh_type: str,
        basis: tp.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Create the complex phase aberration :math:`\exp(\mathrm{i} \sum_j c_j Z_j)` to multiply the pupil with.

    Parameters
    ----------
    zernike_coefficients : torch.Tensor or sequence of floats
        Coefficients :math:`c_j` (peak phase in radians) of the first modes in OSA order. Gradients with respect
        to a tensor of coefficients are propagated.
    n_pix_pupil : int
        Number of pixels of the pupil function.
    mesh_type : str
        Choose 'spherical' or 'cartesian', see :func:`zernike_basis`.
    basis : torch.Tensor, optional
        Precomputed output of ``zernike_basis(len(zernike_coefficients), n_pix_pupil, mesh_type)``, possibly on
        another device. Pass it to avoid rebuilding the basis when only the coefficients change.

    Returns
    -------
    zernike_aberrations : torch.Tensor
        Of dtype `torch.complex64`, of shape `(n_pix_pupil, n_pix_pupil)` for the Cartesian mesh and
        `(n_pix_pupil,)` for the spherical mesh, on the device of `basis` (CPU if not given).

    """
    if not isinstance(zernike_coefficients, torch.Tensor):
        zernike_coefficients = torch.tensor(zernike_coefficients)
    zernike_coefficients = zernike_coefficients.reshape(-1)
    n_modes = zernike_coefficients.shape[0]

    if basis is None:
        basis = zernike_basis(n_modes, n_pix_pupil, mesh_type)
    elif basis.shape[0] != n_modes or basis.shape[-1] != n_pix_pupil:
        raise ValueError(f'The basis of shape {tuple(basis.shape)} does not match {n_modes} coefficients '
                         f'and a pupil of {n_pix_pupil} pixels.')
    if mesh_type == 'spherical':
        _warn_if_not_axisymmetric(zernike_coefficients)

    coefficients = zernike_coefficients.to(basis.device).reshape(-1, *([1] * (basis.ndim - 1)))
    zernike_phase = torch.sum(coefficients * basis, dim=0)
    return torch.exp(1j * zernike_phase).to(torch.complex64)


def create_special_pupil(n_pix_pupil: int, mask = None, tophat_radius: float = 0.5) -> torch.Tensor:
    """
    Special phase masks not included in the space spanned by the Zernike polynomials.

    The supported special phase masks are:
    - None <-> flat phase, Gaussian beam
    - `vortex` <-> donut beam
    - `halfmoon-h` <-> horizontal halfmoon beam
    - `halfmoon-v` <-> vertical halfmoon beam
    - `tophat` <-> tophat beam

    Notes
    -----
    These special masks only applies in the Cartesian case.

    Parameters
    ----------
    n_pix_pupil : int
        Number of pixels on the pupil plane.
    mask : str or torch.Tensor, optional
        Name of the special phase mask (None, 'vortex', 'halfmoon-h', 'halfmoon-v',
        'tophat'), or a custom 2D phase tensor of shape (n_pix_pupil, n_pix_pupil).
    tophat_radius : float
        Radius of the tophat mask. Default is 0.5. TODO: relate to cutoff frequency of the system.

    Returns
    -------
    pupil : torch.Tensor
        Pupil function of the special phase mask.

    """
    kx, ky = create_pupil_mesh(n_pixels=n_pix_pupil)
    if mask is None:
        phase_mask = torch.zeros(n_pix_pupil, n_pix_pupil)
    elif isinstance(mask, torch.Tensor):
        if mask.shape != (n_pix_pupil, n_pix_pupil):
            raise ValueError(f"Custom phase mask must be a 2D Tensor of shape ({n_pix_pupil}, {n_pix_pupil}).")
        phase_mask = mask
    elif mask == 'vortex':
        phase_mask = torch.atan2(kx, ky)
    elif mask == 'halfmoon-h':
        phase_mask = torch.zeros(n_pix_pupil, n_pix_pupil)
        phase_mask[0: n_pix_pupil // 2, :] = torch.pi
    elif mask == 'halfmoon-v':
        phase_mask = torch.zeros(n_pix_pupil, n_pix_pupil)
        phase_mask[:, 0: n_pix_pupil // 2] = torch.pi
    elif mask == 'tophat':
        inner_disk = kx ** 2 + ky ** 2 - tophat_radius ** 2
        phase_mask = torch.where(inner_disk > 0, torch.pi, 0)
    else:
        raise ValueError(f"Invalid mask value {mask}. Must be None, a valid string, or a custom tensor")
    pupil = torch.exp(1j * phase_mask).to(torch.complex64)
    return pupil
