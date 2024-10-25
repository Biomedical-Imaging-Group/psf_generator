"""
Tests for Bessel functions.

"""
import sys

import torch
from torch.autograd import gradcheck

sys.path.append('../..')
from src.psf_generator.utils.bessel import BesselJ0, BesselJ1


def test_bessel_functions():
    inputs = (torch.randn(20, 20, dtype=torch.double, requires_grad=True),)

    j0 = BesselJ0.apply
    assert(gradcheck(j0, inputs, eps=1e-8, atol=1e-8,
                    check_grad_dtypes=True,
                    check_forward_ad=True,
                    check_backward_ad=True,
                    check_batched_forward_grad=True,
                    check_batched_grad=True)), "BesselJ0 does not pass derivative test!"

    j1 = BesselJ1.apply
    assert(gradcheck(j1, inputs, eps=1e-8, atol=1e-8,
                    check_grad_dtypes=True,
                    check_forward_ad=True,
                    check_backward_ad=True,
                    check_batched_forward_grad=True,
                    check_batched_grad=True)), "BesselJ1 does not pass derivative test!"
