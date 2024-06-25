'''
Implements several Newton-Cotes quadrature formulas for performing 1D numerical integration:

`I(x) \approx \int_{a}^{b} f(x) dx`

The integrand is evaluated at N equally-spaced points on [a, b], resulting in a stepsize of 
h = 1 / (N - 1). To allow multiple integrals to be evaluated in parallel, we vectorize the 
integration along dimension `dim = 1`. 

Inputs:
- fs: torch.tensor, shape = (N, num_integrals). The integrand evaluations.
- dx: float. Bin width, also known as h = 1 / (N - 1). The stepsize.

Outputs:
- integrals: torch.tensor, shape = (num_integrals,). The evaluated integrals.
'''

import torch
import warnings

def is_power_of_two(k):
    k = int(k)
    return (k & (k - 1) == 0) and k != 0


def riemann_rule(fs, dx):
    '''
    Riemann quadrature rule, O(h).
    '''
    return torch.sum(fs, dim=0) * dx

def trapezoid_rule(fs, dx):
    '''
    Riemann quadrature rule, O(h ** 2).
    '''
    return 0.5 * (fs[0] + 2.0 * torch.sum(fs[1:-1], dim=0) + fs[-1,:]) * dx

def simpsons_rule(fs, dx):
    '''
    Riemann quadrature rule, O(h ** 4).
    '''
    return (fs[0] + 2 * torch.sum(fs[1:-1], dim=0) + 2 * torch.sum(fs[1:-1:2], dim=0) + fs[-1]) * dx / 3.0

def richard1_rule(fs, dx):
    '''
    Romberg integration truncated at 1 step; equivalent to Simpson's rule, O(h ** 4), 
    when the grid size is set appropriately.

    Warning: this method only achieves higher-order convergence when the number of grid 
    points is N == 2**K + 1.
    '''
    if not(is_power_of_two(fs.shape[0] - 1)):
        warnings.warn("Warning: pupil shape is not of the form (2 ** K + 1)! The computed \
                      integral will not have high-order accuracy.")

    I0 = trapezoid_rule(fs, dx)
    I1 = trapezoid_rule(fs[::2], dx*2)
    return I0 + (I0 - I1) / 3.0

def richard2_rule(fs, dx):
    '''
    Romberg integration truncated at 2 steps; equivalent to two levels of Richardson 
    extrapolation. O(h ** 6).

    Warning: this method only achieves higher-order convergence when the number of grid 
    points is N == 2**K + 1.
    '''
    if not(is_power_of_two(fs.shape[0] - 1)):
        warnings.warn("Warning: pupil shape is not of the form (2 ** K + 1)! The computed \
                      integral will not have high-order accuracy.")

    I0 = trapezoid_rule(fs, dx)
    I1 = trapezoid_rule(fs[::2], dx*2)
    I2 = trapezoid_rule(fs[::4], dx*4)
    I00 =  I0 + (I0 - I1) / 3.0
    I01 =  I1 + (I1 - I2) / 3.0
    return I00 + (I00 - I01) / 15.0

def richard3_rule(fs, dx):
    '''
    Romberg integration truncated at 3 steps; equivalent to three levels of Richardson 
    extrapolation. O(h ** 8).

    Warning: this method only achieves higher-order convergence when the number of grid 
    points is N == 2**K + 1.
    '''
    if not(is_power_of_two(fs.shape[0] - 1)):
        warnings.warn("Warning: pupil shape is not of the form (2 ** K + 1)! The computed \
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
