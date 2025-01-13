# Copyright Biomedical Imaging Group, EPFL 2025

"""
A collection of functions related to Zernike polynomials.

"""
import warnings

import numpy as np
import torch
from scipy.special import binom
from zernikepy import zernike_polynomials


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


def zernike_nl(n: int, l: int, rho: torch.float, phi: float, radius: float = 1) -> torch.Tensor:
    """
    Compute the Zernike polynomial of order n and m in the polar coordinates

    Parameters
    ----------
    n : int
        Index `n` in the definition on wikipedia, positive integer.
    l : int
        :math:`|l| = m`, `m` is the index m in the definition on wikipedia. `l` can be positive or negative.
    rho : torch.Float
        Radial distance.
    phi : float
        Azimuthal angle.
    radius : float
        Radius of the disk on which the Zernike polynomial is defined, default is 1.

    Returns
    -------
    Z: torch.Tensor
        Zernike polynomial Z(rho, phi) evaluated at `rho` and `phi` given indices `n` and `l`.

    """
    m = abs(l)
    R = 0
    for k in np.arange(0, (n - m) / 2 + 1):
        R = R + (-1) ** k * binom(n - k, k) * binom(n - 2 * k, (n - m) / 2 - k) * (rho / radius) ** (n - 2 * k)

    # radial part
    Z = torch.where(rho <= radius, R, 0)

    # angular part
    Z *= np.cos(m * phi) if l >= 0 else np.sin(m * phi)
    return Z


def index_to_nl(index: int) -> tuple[int, int]:
    """
    Find the [n, l]-pair given OSA index l for Zernike polynomials.

    The OSA index 'j' is defined as :math:`j = (n(n + 2) + l) / 2`.

    Parameters
    ----------
    index : int
        OSA index j.

    Returns
    -------
    (n, - n + 2 * l) : Tuple[int, int]
        Corresponding (n, l)-pair.

    """
    n = 0
    while True:
        for l in range(n + 1):
            if n * (n + 1) / 2 + l == index:
                return n, - n + 2 * l
            elif n * (n + 1) / 2 + l > index:
                raise ValueError('Index out of bounds.')
        n += 1


def create_zernike_aberrations(zernike_coefficients: torch.Tensor, n_pix_pupil: int, mesh_type: str) -> torch.Tensor:
    """
    Create Zernike aberrations for the pupil function.

    Arbitrary Zernike aberrations can be applied to the Cartesian propagator.

    How it works:
    - Given the Zernike coefficients as a 1D Tensor of length `n_zernike`, a stack of the first `n_zernike`
    Zernike polynomials are constructed.
    - Then, the coefficients and the polynomials are multiplied and summed accordingly to create a phase mask.
    Finally, we create the complex field to be multiple with the existing pupil function to add this aberration.

    For the Spherical case, only the axis-symmetric Zernike polynomials (i.e. only dependent on the radius `rho` not the
    angle `phi`), such as _defocus_ and 'primary spherical', can be applied due to the axis-symmetric assumption of the
    spherical propagator. See `Spherical_propagators.py` for details.

    Parameters
    ----------
    zernike_coefficients : torch.Tensor
        1D Tensor of Zernike coefficients
    n_pix_pupil : int
        Number of pixels of the pupil function
    mesh_type : str
        Choose 'spherical' or 'cartesian'.

    Returns
    -------
    Zernike_aberrations: torch.Tensor
        Of type torch.complex64.

    """
    n_zernike = len(zernike_coefficients)
    if mesh_type == 'cartesian':
        zernike_basis = zernike_polynomials(mode=n_zernike-1, size=n_pix_pupil, select='all')
        zernike_coefficients = zernike_coefficients.reshape(1, 1, n_zernike)
        zernike_phase = torch.sum(zernike_coefficients * torch.from_numpy(zernike_basis), dim=2)
    elif mesh_type == 'spherical':
        rho = torch.linspace(0, 1, n_pix_pupil)
        phi = 0
        zernike_phase = torch.zeros(n_pix_pupil)
        for i in range(n_zernike):
            n, l = index_to_nl(index=i)
            curr_coefficient = zernike_coefficients[i]
            if l != 0 and curr_coefficient != 0:
                warnings.warn("Warning: Zernike polynomials that are not axis-symmetric \
                                are not supported in spherical coordinates!")
            elif l == 0:
                zernike_phase += curr_coefficient * zernike_nl(n=n, l=l, rho=rho, phi=phi)
    else:
        raise ValueError(f"Invalid mesh type {mesh_type}, choose 'spherical' or 'cartesian'.")

    return torch.exp(1j * zernike_phase).to(torch.complex64)


def create_special_pupil(n_pix_pupil: int, name: str = 'flat', tophat_radius: float = 0.5) -> torch.Tensor:
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
    name : str
        Name of the special phase mask. Valid choices: None, 'vortex', 'halfmoon-h', 'halfmoon-v', 'tophat'.
    tophat_radius : float
        Radius of the tophat mask. Default is 0.5. TODO: relate to cutoff frequency of the system.

    Returns
    -------
    pupil : torch.Tensor
        Pupil function of the special phase mask.

    """
    valid_names = [None, 'vortex', 'halfmoon-h', 'halfmoon-v', 'tophat']
    kx, ky = create_pupil_mesh(n_pixels=n_pix_pupil)
    if name is None:
        phase_mask = torch.zeros(n_pix_pupil, n_pix_pupil)
    elif name == 'vortex':
        phase_mask = torch.atan2(kx, ky)
    elif name == 'halfmoon-h':
        phase_mask = torch.zeros(n_pix_pupil, n_pix_pupil)
        phase_mask[0: n_pix_pupil // 2, :] = torch.pi
    elif name == 'halfmoon-v':
        phase_mask = torch.zeros(n_pix_pupil, n_pix_pupil)
        phase_mask[:, 0: n_pix_pupil // 2] = torch.pi
    elif name == 'tophat':
        inner_disk = kx ** 2 + ky ** 2 - tophat_radius ** 2
        phase_mask = torch.where(inner_disk > 0, torch.pi, 0)
    else:
        raise ValueError(f'Invalid name for the special pupil {name}, choose one of the following: {valid_names}')
    pupil = torch.exp(1j * phase_mask).to(torch.complex64)
    return pupil
