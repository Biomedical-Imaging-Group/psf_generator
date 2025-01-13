# Copyright Biomedical Imaging Group, EPFL 2025

"""
A collection of custom 2D FFT functions.

- `custom_fft2`: custom 2D FFT
- `custom_ifft2`: custom 2D inverse FFT
- `czt1d`: 1D chirp Z-transform
- `czt2d`: 2D chirp Z-transform

"""

__all__ = ['custom_fft2', 'custom_ifft2']

import typing as tp

import torch
from torch.fft import fft, fft2, ifft, ifft2


def _validate_shape(shape_in: tp.Tuple, shape_out: tp.Optional[tp.Tuple]) -> tp.Tuple:
    """
    Validate the shape of the input and output.

    Parameters
    ----------
    shape_in : sequence of ints
        Shape of the last two dimension of the input image.
    shape_out : sequence of ints or None
        Shape of the output image.

    Returns
    -------
    shape_out : tuple
        processed output shape

    """
    if shape_in[0] != shape_in[1]:
        raise ValueError('The input image must be square!')
    elif shape_out is None:
        shape_out = shape_in
    K, L = shape_out[-2:]
    if K != L:
        print('Warning: Output of different size in each dimension; enforcing squared output.')
        shape_out = (max(K, L), max(K, L))
    return shape_out

def _create_w_phase(start: float, end: float, steps: int, include_end: bool) -> float:
    """
    Create the W factor.

    Parameters
    ----------
    start : float
        start point of the sampling
    end : float
        end point of the sampling
    steps : int
        number of sampling steps
    include_end : bool
        whether to include the end point

    Returns
    -------
    w_phase : float
        W factor
    """
    points = steps - 1 if include_end else steps
    w_phase = (end - start) / points
    return w_phase

def _apply_fftshift(x: torch.Tensor, shape_out: tp.Optional[tp.Tuple], k_start: float,
                    K: int, N: int, w_phase: float, a_phase: float) -> torch.Tensor:
    """
    Apply fftshift on the input image.

    Parameters
    ----------
    x : torch.Tensor
        Input 2D image.
    shape_out : tuple or None
        shape of the output image.
    k_start : float
        start point of sampling.
    K : int
        size of the output image.
    N : int
        size of the input image.
    w_phase : float
        W factor
    a_phase : float
        A factor.

    Returns
    -------
    output: torch.Tensor
        fftshifted image.
    """
    k = torch.arange(K)
    kx, ky = torch.meshgrid(k, k, indexing='ij')
    center_correction = torch.exp(1j * (N - 1) / 2 * (2 * k_start - w_phase * (kx + ky))).to(x.device)
    return czt2d(x, shape_out, w_phase, a_phase) * center_correction


def custom_fft2(x: torch.Tensor, shape_out=None, k_start: float = 0.0, k_end: float = 2 * torch.pi,
                norm: str = 'ortho', fftshift_input: bool = False, include_end: bool = False) -> torch.Tensor:
    r"""
    Custom 2D FFT that allows to zoom on the region of interest in the Fourier plane.

    The output image is square.

    Parameters
    ----------
    x : torch.Tensor
        Input square image
    shape_out : sequence of ints, optional
        Shape of the output image.
        If None, same as the shape of the input.
    k_start : float
        Start point of sampling on the complex circle. Default is `0.0`.
    k_end : float
        End point of sampling on the complex circle. Default is :math:`2\pi`.
    norm : {"ortho", "forward", "backward"}, optional
        Normalization mode. Default is "ortho".
    fftshift_input : bool, optional
        Whether to apply fftshift on the input image. Default is "False".
    include_end : bool, optional
        Whether to include the end point of sampling. Default is "False".

    Returns
    -------
    output : torch.Tensor
        Custom 2D FFT of the input image.

    """
    shape_out = _validate_shape(x.shape[-2:], shape_out)
    K = shape_out[0]
    N = x.shape[-2]

    w_phase = _create_w_phase(k_start, k_end, K, include_end)
    a_phase = k_start

    if fftshift_input:
        result = _apply_fftshift(x, shape_out, k_start, K, N, w_phase, a_phase)
    else:
        result = czt2d(x, shape_out, w_phase, a_phase)

    if norm =='ortho':
        return result / K
    elif norm == 'forward':
        return result / K**2
    elif norm == 'backward':
        return result


def custom_ifft2(x: torch.Tensor, shape_out=None, k_start: float = 0.0, k_end: float = 2 * torch.pi,
                norm: str = 'ortho', fftshift_input: bool = False, include_end: bool = False) -> torch.Tensor:
    r"""
    Custom 2D inverse FFT that allows to zoom on the region of interest in the Fourier plane.

    The output image is square.

    Parameters
    ----------
    x : torch.Tensor
        Input square image.
    shape_out : sequence of ints, optional
        Shape of the output image.
        If None, same as the shape of the input.
    k_start : float
        Start point of sampling on the complex circle. Default is `0.0`.
    k_end : float
        End point of sampling on the complex circle. Default is :math:`2\pi`.
    norm : {"ortho", "forward", "backward"}, optional
        Normalization mode. Default is "ortho".
    fftshift_input : bool, optional
        Whether to apply fftshift on the input image. Default is "False".
    include_end : bool, optional
        Whether to include the end point of sampling. Default is "False".

    Returns
    -------
    output : torch.Tensor
        Custom 2D inverse FFT of the input image.

    """
    shape_out = _validate_shape(x.shape[-2:], shape_out)
    K = shape_out[0]
    N = x.shape[-2]

    w_phase = _create_w_phase(k_start, k_end, K, include_end)
    a_phase = - k_start

    if fftshift_input:
        result = _apply_fftshift(x, shape_out, k_start, K, N, w_phase, a_phase)
        angle = 2 * (N - 1) * k_end
        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle)
        result *= torch.exp(1j * angle)
    else:
        result = czt2d(x, shape_out, w_phase, a_phase)

    if norm =='ortho':
        return result / K
    elif norm == 'forward':
        return result
    elif norm == 'backward':
        return result / K**2


def czt1d(x: torch.Tensor, shape_out=None, w_phase=None, a_phase: torch.Tensor = 0.0) -> torch.Tensor:
    """
    1D chirp Z-transform implemented in PyTorch.

    Parameters
    ----------
    x : torch.Tensor
        Input 1D image.
    shape_out : int, optional
        Length of the output image.
        If None, same as the length of the input.
    w_phase : torch.Tensor, optional
        W factor in the definition of the CZT.
    a_phase : torch.Tensor, optional
        A factor in the definition of the CZT

    Returns
    -------
    output: torch.Tensor
        CZT of the input 1D image.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.CZT.html

    """
    shape_in = x.shape[-1]
    if shape_out is None:
        shape_out = shape_in
    if w_phase is None:
        w_phase = - 2 * torch.pi / shape_out
    max_dim = max(shape_in, shape_out)
    fft_dim = int(2 ** torch.ceil(torch.log2(torch.tensor(shape_in + shape_out - 1))))
    device = x.device

    k = torch.arange(max_dim, device=device)
    wk2 = torch.exp(1j * w_phase * k ** 2 / 2).to(device)
    aw_factor = torch.exp(- 1j * a_phase * k[:shape_in]).to(device) * wk2[:shape_in]
    second_factor = fft(
        1 / torch.hstack((torch.flip(wk2[1:shape_in], dims=(0,)), wk2[:shape_out])), fft_dim)
    idx = slice(shape_in - 1, shape_in + shape_out - 1)

    output = ifft(fft(x * aw_factor, n=fft_dim) * second_factor)
    output = wk2[:shape_out] * output[..., idx]

    return output


def czt2d(x: torch.Tensor, shape_out=None, w_phase=None, a_phase: float = 0.0) -> torch.Tensor:
    """
    2D chirp Z-transform implemented in PyTorch.

    Parameters
    ----------
    x : torch.Tensor
        Input 2D image.
    shape_out : sequence of ints, optional
        Shape of the output image.
        If None, same as the length of the input.
    w_phase : float, optional
        W factor in the definition of the CZT.
    a_phase : float, optional
        A factor in the definition of the CZT

    Returns
    -------
    output: torch.Tensor
        CZT of the input 2D image.

    """
    shape_out = _validate_shape(x.shape[-2:], shape_out)
    K = shape_out[0]
    N = x.shape[-2]

    if w_phase is None:
        w_phase = - 2 * torch.pi / K
    max_dim = max(N, K)
    fft_dim = int(2 ** torch.ceil(torch.log2(torch.tensor(N + K - 1))))
    device = x.device

    k = torch.arange(max_dim).to(device)
    kx, ky = torch.meshgrid(k, k, indexing='ij')

    wk2 = torch.exp(1j * w_phase * (kx**2+ky**2) / 2).to(device)
    aw_factor = torch.exp(- 1j * a_phase * (kx[:N, :N]+ky[:N, :N])).to(device) * wk2[:N, :N]
    second_factor = fft2(1 / torch.hstack(
        (torch.vstack(
            (torch.flip(wk2[1:N, 1:N], dims=(0,1)),
            torch.flip(wk2[:K, 1:N], dims=(1,))),
        ),
        torch.vstack(
            (torch.flip(wk2[1:N, :K], dims=(0,)),
            wk2[:K, :K]),
        )),
    ), s=(fft_dim,fft_dim))
    idx = slice(N - 1, N + K - 1)

    output = ifft2(fft2(x * aw_factor, s=(fft_dim,fft_dim)) * second_factor)
    output = wk2[:K, :K] * output[..., idx, idx]

    return output

