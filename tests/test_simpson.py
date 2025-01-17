"""
Test self-implemented PyTorch version of trapezoid and simpson's integration method against the Scipy ones.

"""
import numpy as np
import torch
from scipy.integrate import simpson, trapezoid
from scipy.special import j0

from psf_generator.utils.integrate import simpsons_rule, trapezoid_rule

a = 0
b = np.pi/4
N = 201
dx = (b - a) / (N - 1)
x = np.linspace(a, b, N)
y = np.sin(x) * np.cos(x) * j0(np.sin(x))

def test_simpson():
    ref = simpson(y=y, dx=dx)
    test = simpsons_rule(torch.tensor(y), dx=dx).detach().cpu().numpy()
    np.testing.assert_almost_equal(ref, test)

def test_trapezoid():
    ref = trapezoid(y=y, dx=dx)
    test = trapezoid_rule(torch.tensor(y), dx=dx).detach().cpu().numpy()
    np.testing.assert_almost_equal(ref, test)
