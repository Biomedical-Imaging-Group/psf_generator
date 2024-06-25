"""
Contains AD-enabled overrides for the bessel functions `bessel_j0` and `bessel_j1`;
torch's implementations do not have gradient tracking as of v1.13.1.
"""
import torch
from typing import Any
from torch.autograd import Function, gradcheck
from torch.special import bessel_j0 # as __bessel_j0
from torch.special import bessel_j1 # as __bessel_j1

class BesselJ0(Function):
    '''
    Differentiable version of `bessel_j0(x)`.
    '''
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.save_for_forward(x)
        return bessel_j0(x)
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def vjp(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        '''
        Vector-Jacobian product, for reverse-mode AD (`backward()`).
        '''
        x, = ctx.saved_tensors
        return -bessel_j1(x) * grad_output
    
    @staticmethod
    def jvp(ctx: Any, grad_input: torch.Tensor) -> torch.Tensor:
        '''
        Jacobian-vector product, for forward-mode AD.
        '''
        x, = ctx.saved_tensors
        return -bessel_j1(x) * grad_input
    

class BesselJ1(Function):
    '''
    Differentiable version of `bessel_j1(x)`.
    '''
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        result = bessel_j1(x)
        ctx.save_for_backward(x, result)
        ctx.save_for_forward(x, result)
        return result
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def vjp(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        '''
        Vector-Jacobian product, for reverse-mode AD (`backward()`).
        '''
        x, j1 = ctx.saved_tensors
        j1_norm_x = torch.where(x == 0.0, 0.5, j1 / x)
        jac = bessel_j0(x) - j1_norm_x
        return jac * grad_output
    
    @staticmethod
    def jvp(ctx: Any, grad_input: torch.Tensor) -> torch.Tensor:
        '''
        Jacobian-vector product, for forward-mode AD.
        '''
        x, j1 = ctx.saved_tensors
        j1_norm_x = torch.where(x == 0.0, 0.5, j1 / x)
        jac = bessel_j0(x) - j1_norm_x
        return jac * grad_input


if __name__ == 'main':
    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True))

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
