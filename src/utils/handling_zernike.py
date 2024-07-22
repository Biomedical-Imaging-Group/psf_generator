from typing import List, Tuple

import numpy as np
import torch
from scipy.special import binom
from zernikepy import zernike_polynomials


def create_pupil_mesh(n_pixels: int) -> Tuple[torch.Tensor, ...]:
    """
    Create a 2D square meshgrid for the pupil function.
    Parameters
    ----------
    n_pixels: int
        number of pixels for the pupil function
    Returns
    -------
    (kx, ky): Tuple[torch.Tensor, ...]
        two Tensors that represnt the 2D coordinates on the mesh
    """
    x = torch.linspace(-1, 1, n_pixels)
    y = torch.linspace(-1, 1, n_pixels)
    kx, ky = torch.meshgrid(x, y, indexing='xy')
    return kx, ky


def zernike_nl(n: int, l: int, rho: torch.float, phi: float, radius: float = 1) -> torch.Tensor:
    """
    Computation of the Zernike polynomial of order n and m in the polar coordinates

    Parameters
    ----------
    n: int
        index n in the definition on wikipedia, positive integer
    l: int
        |l| = m, m is the index m in the definition on wikipedia. l can be positive or negative
    rho: torch.Float
        radial distance
    phi: float
        azimuthal angle
    radius: float
        radius of the disk on which the Zernike polynomial is defined, default is 1

    Returns
    -------
    Z: torch.Tensor
        Zernike polynomial Z(rho, phi) evaluated at 'rho' and 'phi' given indices n and l
    """
    m = abs(l)
    R = 0
    for k in np.arange(0, (n - m) / 2 + 1):
        R = R + (-1) ** k * binom(n - k, k) * binom(n - 2 * k, (n - m) / 2 - k) * (rho / radius) ** (n - 2 * k)

    # radial part
    Z = torch.where(torch.tensor(rho <= radius), R, 0)

    # angular part
    Z *= np.cos(m * phi) if l >= 0 else np.sin(m * phi)
    return Z


def index_to_nl(index: int) -> Tuple[int, int]:
    """
    Find the [n, l]-pair given OSA index l for Zernike polynomials.
    The OSA index 'j' is defined as
    $j = (n(n + 2) + l) / 2$.

    Parameters:
    ----------
    index: int
        OSA index j

    Returns
    -------
    [n, - n + 2 * l]: [int, int]
        Corresponding [n, l]-pair
    """
    n = 0
    while True:
        for l in range(n + 1):
            if n * (n + 1) / 2 + l == index:
                return n, - n + 2 * l
            elif n * (n + 1) / 2 + l > index:
                raise ValueError('Index out of bounds.')
        n += 1


def create_zernike_aberrations(zernike_coefficients: List[torch.Tensor], n_pix_pupil: int, mesh_type: str) -> torch.Tensor:
    """
    Create Zernike aberrations for the pupil function in the Cartesian case.
    For Scalar or Vectorial Cartesian cases, Zernike abberations can be added to the pupil function.
    Given the Zernike coefficients as a 1D Tensor of length `n_zernike`, a stack of the first `n_zernike`
    Zernike polymonials are constructed.
    Then, the coefficients and the polymonials are multiplied and summed accordingly to create a phase mask.
    Finally, we create the complex field to be multiplide with the existing  pupil function to add this aberration.

    Parameters
    ----------
    zernike_coefficients: torch.Tensor
        1D Tensor of Zernike coefficients
    n_pix_pupil: int
        number of pixels of the pupil function
    mesh_type: str
        choose `polar` or 'cartesian'.
    Returns
    -------
    torch.Tensor of type torch.complex64
    """
    n_zernike = len(zernike_coefficients)
    if mesh_type == 'cartesian':
        zernike_basis = zernike_polynomials(mode=n_zernike-1, size=n_pix_pupil, select='all')
        zernike_coefficients = torch.tensor(zernike_coefficients).reshape(1, 1, n_zernike)
        zernike_phase = torch.sum(zernike_coefficients * torch.from_numpy(zernike_basis), dim=2)
    elif mesh_type == 'polar':
        rho = torch.linspace(0, 1, n_pix_pupil)
        phi = 0
        zernike_phase = torch.zeros(n_pix_pupil)
        for i in range(n_zernike):
            n, l = index_to_nl(index=i)
            curr_coef = zernike_coefficients[i]
            if l != 0 and curr_coef != 0:
                print("Warning: Zernike coefficients for l != 0 are not supported in polar coordinates.")
            elif l == 0:
                zernike_phase += curr_coef * torch.tensor(zernike_nl(n=n, l=l, rho=rho, phi=phi))
    else:
        raise ValueError(f"Invalid mesh type {mesh_type}, choose 'polar' or 'cartesian'.")

    return torch.exp(1j * zernike_phase).to(torch.complex64)