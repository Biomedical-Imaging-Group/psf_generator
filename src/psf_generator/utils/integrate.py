# Copyright Biomedical Imaging Group, EPFL 2025

r"""
A collection of numerical integration rules in 1D.

We provide two common rules written in PyTorch: `trapezoid` and `simpson`, equivalent to their counterparts in
`scipy.integrate`.

How it works is briefly explained here:

We would like to compute the integral :math:`I = \int_{a}^{b} f(x) dx` of a function :math:`f(x)` over an interval
:math:`[a, b]`.
We partition :math:`[a, b]` into :math:`n` equidistant sub-intervals by :math:`N=n+1` nots
:math:`a \leq x_0 < x_1 < \ldots < x_n \leq b`, in other words, :math:`x_k = a + kh, k=0, \ldots, n` with stepsize
:math:`h=\frac{b - a}{n}`.

The composite trapezoid rule

.. math::  T_n = \frac{h}{2}\left(f(a) + 2\sum_{k=1}^{n-1}f(x_k) + f(b)\right).

The composite Simpson's rule

.. math:: S_n = \frac{h}{6}\left(f(a) + 4\sum_{k=1}^{n-1}f{x_{k+\frac{1}{2}}} + 2\sum_{k=1}^{n-1}f(x_k) + f(b)\right),

where :math:`x_{k+\frac{1}{2}} = \frac{x_k + x_{k+1}}{2}`.

In implementation, trapezoid is written exactly as its formula.
Simpson's rule requires the function value of the midpoint which is not provided, we view the partition stepsize as
:math:`h = 2\frac{b - a}{n}` and the odd nots as the midpoint instead.
The formula then becomes

.. math:: S_n = \frac{h}{3}\left(f(a) + 4\sum_{k=1}^{n/2}f(x_{2k-1}) + 2\sum_{k=1}^{n/2-1}f(x_{2k}) + f(b)\right).

"""

__all__ = ['riemann_rule', 'simpsons_rule']

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
    r"""
    Composite trapezoid rule, see also [1]_.

    Parameters
    ----------
    fs : torch.Tensor
        The integrand evaluations of shape (N, number_of_integrals).
    dx : float
        Step size.

    Returns
    -------
    output: torch.Tensor
        Integral evaluated by trapezoid rule of shape (num_integrals,).

    Notes
    -----
    - :math:`h = \frac{b - a}{N - 1}`.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html#scipy.integrate.trapezoid

    """
    return 0.5 * (fs[0] + 2.0 * torch.sum(fs[1:-1], dim=0) + fs[-1]) * dx

def simpsons_rule(fs: torch.Tensor, dx: float) -> torch.Tensor:
    r"""
    Composite Simpson's rule, see also [2]_.

    Parameters
    ----------
    fs : torch.Tensor
        The integrand evaluations of shape (N, number_of_integrals).
    dx : float
        Step size.

    Returns
    -------
    output: torch.Tensor
        Integral evaluated by Simpson's rule of shape (num_integrals,).

    Notes
    -----
    - :math:`h = \frac{b - a}{N - 1}`.
    - Simpson's rule only works correctly with grids of odd sizes (i.e. :math:`N = 2^K + 1`).

    References
    ----------
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html#scipy.integrate.simpson

    """
    if fs.shape[0] % 2 == 0:
        warnings.warn("Pupil size is not an odd number! The computed \
                      integral will not have high-order accuracy.")
    return (fs[0] + 4 * torch.sum(fs[1:-1:2], dim=0) + 2 * torch.sum(fs[2:-1:2], dim=0) + fs[-1]) * dx / 3.0
