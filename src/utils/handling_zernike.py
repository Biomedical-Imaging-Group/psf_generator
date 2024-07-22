import numpy as np
import torch
from scipy.special import binom


def zernike_nl(n: int, l: int, rho: torch.float, phi: float, radius: float = 1):
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


def index_to_nl(index: int):
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
