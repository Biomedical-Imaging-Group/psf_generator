import torch
import numpy as np
import matplotlib.pyplot as plt
from .pupil import ScalarPolarPupil, ScalarCartesianPupil

def get_polar_ck(N: int) -> np.ndarray:
    """
    Generate random Zernike coefficients for a radially symmetric pupil field.

    Inputs:
    - N: int. Number of nonzero terms.
    """
    idxs = np.arange(N)
    idxs = 2 * (idxs + 1) * idxs

    values = (2.0 * np.random.rand(N) - 1.0)
    ck = np.zeros(idxs[-1] + 1)
    ck[idxs] = values
    return ck

def test_pupils(Nr=128, zernike_order=3, plot=False):
    """
    Numerically evaluate the aberration fields of a ScalarPolar and ScalarCartesian pupil
    using the same set of Zernike coefficients. The two E fields should be identical up to
    numerical precision.
    """
    Nr = 128
    ck = get_polar_ck(zernike_order)

    field_P = ScalarPolarPupil(
        n_pix_pupil = 1 + Nr,
        zernike_coefficients=ck).field.squeeze()
    field_C = ScalarCartesianPupil(
        n_pix_pupil = 1 + Nr*2,
        zernike_coefficients=ck).field.squeeze()

    assert torch.allclose(field_P, field_C[Nr,Nr:]), "Polar and cartesian fields do not match!"

    if plot:
        plt.figure()
        plt.subplot(211)
        plt.plot(field_P.real, label="Re(E), Polar")
        plt.plot(field_C[Nr,Nr:].real, label="Re(E), Cartesian")
        plt.legend()
        plt.subplot(212)
        plt.plot(field_P.imag, label="Im(E), Polar")
        plt.plot(field_C[Nr,Nr:].imag, label="Im(E), Cartesian")
        plt.legend()

        plt.tight_layout()


if __name__ == "__main__":
    for _ in range(10):
        test_pupils()
