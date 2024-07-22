import numpy as np
import torch
from torch.fft import fft, fft2, ifft, ifft2


def custom_fft2(x, shape_out=None, k_start=0, k_end=2*np.pi,
                norm='ortho', fftshift_input=False, include_end=False):
    shape_in = x.shape
    N, M = shape_in[-2:]
    if shape_out is None:
        shape_out = shape_in
    K, L = shape_out[-2:]
    if K != L:
        print('Warning: Output dimensions are different; enforcing squared output.')
        K, L = max(K,L), max(K,L)

    if include_end:
        w_phase = - (k_end - k_start) / (K-1)
    else:
        w_phase = - (k_end - k_start) / K
    a_phase = k_start

    if fftshift_input:
        k = torch.arange(K)
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        center_correction = torch.exp(1j * (N - 1) / 2 * (2*k_start - w_phase * (kx+ky))).to(x.device)
        result = czt2d(x, shape_out, w_phase, a_phase) * center_correction
    else:
        result = czt2d(x, shape_out, w_phase, a_phase)
    if norm =='ortho':
        return result / K
    elif norm == 'forward':
        return result / K**2
    elif norm == 'backward':
        return result


def custom_ifft2(x, shape_out=None, k_start=0, k_end=2*np.pi,
                 norm='ortho', fftshift_input=False, include_end=False):
    shape_in = x.shape
    N, M = shape_in[-2:]
    if shape_out is None:
        shape_out = shape_in
    K, L = shape_out[-2:]
    if K != L:
        print('Warning: Output dimensions are different; enforcing squared output.')
        K, L = max(K,L), max(K,L)

    if include_end:
        w_phase = (k_end - k_start) / (K-1)
    else:
        w_phase = (k_end - k_start) / K
    a_phase = - k_start

    if fftshift_input:
        k = torch.arange(K)
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        center_correction = torch.exp(1j * (N - 1) / 2 * (2*k_start - w_phase * (kx+ky))).to(x.device)
        result = czt2d(x, shape_out, w_phase, a_phase) * center_correction
        angle = 2 * (N - 1) * k_end
        result *= torch.exp(1j*torch.tensor(angle))
    else:
        result = czt2d(x, shape_out, w_phase, a_phase)
    if norm =='ortho':
        return result / K
    elif norm == 'forward':
        return result
    elif norm == 'backward':
        return result / K**2


def czt1d(x, shape_out=None, w_phase=None, a_phase=0):
    shape_in = x.shape[-1]
    if shape_out is None:
        shape_out = shape_in
    if w_phase is None:
        w_phase = - 2 * np.pi / shape_out
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


def czt2d(x, shape_out=None, w_phase=None, a_phase=0):
    shape_in = x.shape
    if shape_out is None:
        shape_out = shape_in
    N, M = shape_in[-2:]
    if N != M:
        print('Warning: Output dimensions are different; enforcing squared output.')
        N, M = max(N,M), max(N,M)
    K, L = shape_out[-2:]
    if K != L:
        print('Warning: Output dimensions are different; enforcing squared output.')
        K, L = max(K,L), max(K,L)

    if w_phase is None:
        w_phase = - 2 * np.pi / K
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
            torch.flip(wk2[:K, 1:N], dims=(1,)))
        ),
        torch.vstack(
            (torch.flip(wk2[1:N, :K], dims=(0,)),
            wk2[:K, :K])
        ))
    ), s=(fft_dim,fft_dim))
    idx = slice(N - 1, N + K - 1)

    output = ifft2(fft2(x * aw_factor, s=(fft_dim,fft_dim)) * second_factor)
    output = wk2[:K, :K] * output[..., idx, idx]

    return output

