# Copyright Biomedical Imaging Group, EPFL 2024

r"""
A collection of Newton-Cotes quadrature rules for numerical integration in 1D.

The definition of the integral :math:`I(x)` of a function :math:`f(x)` over an interval :math:`[a, b]` is

.. math:: I(x) = \int_{a}^{b} f(x) dx

The integrand is evaluated at `N` equally-spaced points on :math:`[a, b]`, resulting in a step size of
:math:`h = \frac{1}{N - 1}`.
We vectorize the integration along dimension `dim = 1` to allow multiple integrals to be evaluated in parallel.
"""

__all__ = ['riemann_rule', 'trapezoid_rule', 'simpsons_rule', 'richard1_rule', 'richard2_rule', 'richard3_rule']

import warnings

import torch


def is_power_of_two(k: int) -> bool:
    """
    Check whether a given integer `k` is a power of 2 and nonzero.

    Return a boolean variable.

    If `k` is not an integer, take the integer part of it.

    Parameters
    ----------
    k : int
        integer to check

    Returns
    -------
    output: bool

    """
    k = int(k)
    return (k & (k - 1) == 0) and k != 0


def riemann_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Riemann quadrature rule of precision :math:`O(h)`.

    Parameters
    ----------
    fs : torch.Tensor
        The integrand evaluations of shape (N, number_of_integrals).
    dx : float
        Bin width or step size for evaluation :math:`h = 1 / (N - 1)`.

    Returns
    -------
    output: torch.Tensor
        Integral evaluated by Riemann rule of shape (num_integrals,).

    """
    return torch.sum(fs, dim=0) * dx

def trapezoid_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Trapezoid rule of precision :math:`O(h^2)`.

    Parameters
    ----------
    fs : torch.Tensor
        The integrand evaluations of shape (N, number_of_integrals).
    dx : float
        Bin width or step size for evaluation :math:`h = 1 / (N - 1)`.

    Returns
    -------
    output: torch.Tensor
        Integral evaluated by trapezoid rule of shape (num_integrals,).

    """
    return 0.5 * (fs[0] + 2.0 * torch.sum(fs[1:-1], dim=0) + fs[-1,:]) * dx

def simpsons_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Simpson's rule of precision :math:`O(h^4)`.

    Parameters
    ----------
    fs : torch.Tensor
        The integrand evaluations of shape (N, number_of_integrals).
    dx : float
        Bin width or step size for evaluation :math:`h = 1 / (N - 1)`.

    Returns
    -------
    output: torch.Tensor
        Integral evaluated by Simpson's rule of shape (num_integrals,).

    Notes
    -----
    Simpson's rule only works correctly with grids of odd sizes (i.e. N == 2*K + 1)!

    """
    if fs.shape[0] % 2 == 0:
        warnings.warn("Pupil size is not an odd number! The computed \
                      integral will not have high-order accuracy.")
    return (fs[0] + 2 * torch.sum(fs[1:-1], dim=0) + 2 * torch.sum(fs[1:-1:2], dim=0) + fs[-1]) * dx / 3.0

def richard1_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Romberg integration truncated at 1 step.

    Equivalent to Simpson's rule with precision :math:`O(h^4)` when the grid size is set appropriately.

    Parameters
    ----------
    fs : torch.Tensor
        The integrand evaluations of shape (N, number_of_integrals).
    dx : float
        Bin width or step size for evaluation :math:`h = 1 / (N - 1)`.

    Returns
    -------
    output: torch.Tensor
        Integral evaluated by Richard's rule of 4th-order precision of shape (num_integrals,).

    Notes
    -----
    This method only achieves higher-order convergence when the number of grid points is N == 2**K + 1.

    """
    if not is_power_of_two(fs.shape[0] - 1):
        warnings.warn("Pupil shape is not of the form (2 ** K + 1)! The computed \
                      integral will not have high-order accuracy.")

    I0 = trapezoid_rule(fs, dx)
    I1 = trapezoid_rule(fs[::2], dx*2)
    return I0 + (I0 - I1) / 3.0

def richard2_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Romberg integration truncated at 2 steps.

    Equivalent to two levels of Richardson extrapolation of precision :math:`O(h^6)`.

    Parameters
    ----------
    fs : torch.Tensor
        The integrand evaluations of shape (N, number_of_integrals).
    dx : float
        Bin width or step size for evaluation :math:`h = 1 / (N - 1)`.

    Returns
    -------
    output: torch.Tensor
        Integral evaluated by Richard's rule of 6th-order precision of shape (num_integrals,).

    Notes
    -----
    This method only achieves higher-order convergence when the number of grid points is N == 2**K + 1.

    """
    if not is_power_of_two(fs.shape[0] - 1):
        warnings.warn("Pupil shape is not of the form (2 ** K + 1)! The computed \
                      integral will not have high-order accuracy.")

    I0 = trapezoid_rule(fs, dx)
    I1 = trapezoid_rule(fs[::2], dx*2)
    I2 = trapezoid_rule(fs[::4], dx*4)
    I00 =  I0 + (I0 - I1) / 3.0
    I01 =  I1 + (I1 - I2) / 3.0
    return I00 + (I00 - I01) / 15.0

def richard3_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    """
    Romberg integration truncated at 3 steps.
    Equivalent to three levels of Richardson extrapolation of precision :math:`O(h^8)`.

    Parameters
    ----------
    fs : torch.Tensor
        The integrand evaluations of shape (N, number_of_integrals).
    dx : float
        Bin width or step size for evaluation :math:`h = 1 / (N - 1)`.

    Returns
    -------
    output: torch.Tensor
        Integral evaluated by Richard's rule of 8th-order precision of shape (num_integrals,).

    Notes
    -----
    This method only achieves higher-order convergence when the number of grid points is N == 2**K + 1.

    """
    if not is_power_of_two(fs.shape[0] - 1):
        warnings.warn("Pupil shape is not of the form (2 ** K + 1)! The computed \
                      integral will not have high-order accuracy.")

    I0 = trapezoid_rule(fs, dx)
    I1 = trapezoid_rule(fs[::2], dx*2)
    I2 = trapezoid_rule(fs[::4], dx*4)
    I3 = trapezoid_rule(fs[::8], dx*8)
    I00 =  I0 + (I0 - I1) / 3.0
    I01 =  I1 + (I1 - I2) / 3.0
    I02 =  I2 + (I2 - I3) / 3.0

    I000 = I00 + (I00 - I01) / 15.0
    I001 = I01 + (I01 - I02) / 15.0

    return I000 + (I000 - I001) / 63.0
