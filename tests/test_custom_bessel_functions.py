import torch
from torch.autograd import gradcheck
import sys
sys.path.append('../..')
from src.utils.bessel import BesselJ0, BesselJ1


def test_bessel_functions():
    input = (torch.randn(20, 20, dtype=torch.double, requires_grad=True))

    j0 = BesselJ0.apply
    assert(gradcheck(j0, input, eps=1e-8, atol=1e-8,
                    check_grad_dtypes=True,
                    check_forward_ad=True,
                    check_backward_ad=True,
                    check_batched_forward_grad=True,
                    check_batched_grad=True)), "BesselJ0 does not pass derivative test!"

    j1 = BesselJ1.apply
    assert(gradcheck(j1, input, eps=1e-8, atol=1e-8,
                    check_grad_dtypes=True,
                    check_forward_ad=True,
                    check_backward_ad=True,
                    check_batched_forward_grad=True,
                    check_batched_grad=True)), "BesselJ1 does not pass derivative test!"
