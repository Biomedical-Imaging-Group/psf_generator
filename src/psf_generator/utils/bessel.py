# Copyright Biomedical Imaging Group, EPFL 2025

"""
A collection of custom Bessel functions with gradient tracking.

These functions contain adjoint-enabled overrides for the PyTorch build-in `bessel_j0` and `bessel_j1` as
those do not have gradient tracking as of v1.13.1.

"""

__all__ =['BesselJ0', 'BesselJ1']

from typing import Any

import torch
from torch.autograd import Function
from torch.special import (
    bessel_j0,  # as __bessel_j0
    bessel_j1,  # as __bessel_j1
)


class BesselJ0(Function):
    """
    Differentiable version of PyTorch's `bessel_j0(x)`.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.save_for_forward(x)
        return bessel_j0(x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def vjp(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Vector-Jacobian product, for reverse-mode adjoint (`backward()`).
        """
        x, = ctx.saved_tensors
        return -bessel_j1(x) * grad_output

    @staticmethod
    def jvp(ctx: Any, grad_input: torch.Tensor) -> torch.Tensor:
        """
        Jacobian-vector product, for forward-mode adjoint.
        """
        x, = ctx.saved_tensors
        return -bessel_j1(x) * grad_input


class BesselJ1(Function):
    """
    Differentiable version of `bessel_j1(x)`.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        result = bessel_j1(x)
        ctx.save_for_backward(x, result)
        ctx.save_for_forward(x, result)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def vjp(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Vector-Jacobian product, for reverse-mode adjoint (`backward()`).
        """
        x, j1 = ctx.saved_tensors
        j1_norm_x = torch.where(x == 0.0, 0.5, j1 / x)
        jac = bessel_j0(x) - j1_norm_x
        return jac * grad_output

    @staticmethod
    def jvp(ctx: Any, grad_input: torch.Tensor) -> torch.Tensor:
        """
        Jacobian-vector product, for forward-mode adjoint.
        """
        x, j1 = ctx.saved_tensors
        j1_norm_x = torch.where(x == 0.0, 0.5, j1 / x)
        jac = bessel_j0(x) - j1_norm_x
        return jac * grad_input
