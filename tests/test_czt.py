"""Tests for the chirp Z-transform based zoom FFT used by the Cartesian propagators."""
import math

import pytest
import torch

from psf_generator.utils.czt import custom_ifft2


def _direct_inverse_transform(x, n_out, k_start, k_end, centred):
    """Evaluate ``X[j1, j2] = sum_m x[m1, m2] exp(i m1' w_j1) exp(i m2' w_j2)`` by brute force.

    ``w_j`` runs from ``k_start`` to ``k_end`` (endpoints included) and ``m'`` is the pixel index
    measured from the first pixel, or from the centre pixel ``(N - 1) / 2`` if ``centred``.
    """
    n_in = x.shape[-1]
    omega = k_start + (k_end - k_start) * torch.arange(n_out, dtype=torch.float64) / (n_out - 1)
    m = torch.arange(n_in, dtype=torch.float64) - ((n_in - 1) / 2 if centred else 0.0)
    kernel = torch.exp(1j * m[:, None] * omega[None, :])  # (n_in, n_out)
    return kernel.T @ x.to(torch.complex128) @ kernel


def test_full_range_matches_torch_ifft2():
    torch.manual_seed(0)
    x = torch.randn(8, 8, dtype=torch.complex64)
    result = custom_ifft2(x, k_start=0.0, k_end=2 * math.pi, norm='backward', include_end=False)
    torch.testing.assert_close(result, torch.fft.ifft2(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('n_in, n_out, k_start, k_end', [
    (9, 7, -1.3, 1.3),   # odd input, symmetric range
    (8, 6, -0.9, 1.4),   # even input, asymmetric range
    (9, 9, 0.2, 2.5),    # one-sided range
])
def test_zoom_of_centred_input_matches_direct_sum(n_in, n_out, k_start, k_end):
    torch.manual_seed(1)
    x = torch.randn(n_in, n_in, dtype=torch.complex64)
    result = custom_ifft2(x, shape_out=(n_out, n_out), k_start=k_start, k_end=k_end,
                          norm='forward', fftshift_input=True, include_end=True)
    expected = _direct_inverse_transform(x, n_out, k_start, k_end, centred=True)
    torch.testing.assert_close(result.to(torch.complex128), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('n_in, n_out, k_start, k_end', [(9, 7, -1.3, 1.3), (8, 6, -0.9, 1.4)])
def test_zoom_of_uncentred_input_matches_direct_sum(n_in, n_out, k_start, k_end):
    torch.manual_seed(2)
    x = torch.randn(n_in, n_in, dtype=torch.complex64)
    result = custom_ifft2(x, shape_out=(n_out, n_out), k_start=k_start, k_end=k_end,
                          norm='forward', fftshift_input=False, include_end=True)
    expected = _direct_inverse_transform(x, n_out, k_start, k_end, centred=False)
    torch.testing.assert_close(result.to(torch.complex128), expected, atol=1e-4, rtol=1e-4)
