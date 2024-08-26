from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from utils.zernike import create_pupil_mesh, create_zernike_aberrations


class Pupil(ABC):
    def __init__(self, n_pix_pupil: int = 128, device: str = 'cpu',
                 zernike_coefficients: torch.Tensor | np.ndarray | list[float] | None = None):
        self.n_pix_pupil = n_pix_pupil
        self.device = device
        if zernike_coefficients is None:
            zernike_coefficients = [0]
        self.zernike_coefficients = torch.tensor(zernike_coefficients)
        self.field = self.initialize_field()
        self.field *= self.zernike_aberrations()

    @abstractmethod
    def initialize_field(self):
        raise NotImplementedError

    @abstractmethod
    def zernike_aberrations(self):
        raise NotImplementedError


class ScalarCartesianPupil(Pupil):
    """
    Define a 2D pupil function for the scalar Cartesian case. The function is defined on the
    unit disk centered at (0,0): u ** 2 + v ** 2 <= 1. The mapping between this domain and
    the physical pupil coordinates are:

        u = sx / s_max
        v = sy / s_max

    such that the physical domain is:

        sx ** 2 + sy ** 2 <= s_max ** 2 = sin(theta_max) ** 2
    """

    def initialize_field(self):
        kx, ky = create_pupil_mesh(n_pixels=self.n_pix_pupil)
        return (kx**2 + ky**2 <= 1).to(torch.complex64).unsqueeze(0).unsqueeze(0).to(self.device)

    def zernike_aberrations(self):
        aberrations = create_zernike_aberrations(self.zernike_coefficients, self.n_pix_pupil, mesh_type='cartesian')
        return aberrations.to(self.device).unsqueeze(0).unsqueeze(0)


class ScalarPolarPupil(Pupil):
    """
    Define a (1D) radial pupil function for the scalar polar case. The function is defined on
    the interval `\rho` \in [0,1]; `\rho` is a "normalized" radius. The conversion to physical
    pupil coordinates - the polar angle `\theta` - is given by:

        \rho = \frac{\sin{\theta}}{\sin{\theta_{max}}}

    such that the physical domain is:

        \theta \leq \theta_{max}
    """

    def initialize_field(self):
        return torch.ones(self.n_pix_pupil).to(torch.complex64).to(self.device).unsqueeze(0).unsqueeze(0)

    def zernike_aberrations(self):
        aberrations = create_zernike_aberrations(self.zernike_coefficients, self.n_pix_pupil, mesh_type='polar')
        return aberrations.to(self.device).unsqueeze(0).unsqueeze(0)


class VectorialCartesianPupil(Pupil):
    def __init__(self, e0x: float = 1.0, e0y: float = 0.0,
                 n_pix_pupil: int = 128, device: str = 'cpu',
                 zernike_coefficients: torch.Tensor | np.ndarray | list[float] | None = None):
        self.e0x = e0x
        self.e0y = e0y
        super().__init__(n_pix_pupil, device, zernike_coefficients)

    def initialize_field(self):
        kx, ky = create_pupil_mesh(n_pixels=self.n_pix_pupil)
        single_field = (kx**2 + ky**2 <= 1).to(torch.complex64)
        return torch.stack((self.e0x * single_field, self.e0y * single_field),
                           dim=0).unsqueeze(0).to(self.device)

    def zernike_aberrations(self):
        aberrations = create_zernike_aberrations(self.zernike_coefficients, self.n_pix_pupil, mesh_type='cartesian')
        return aberrations.to(self.device).unsqueeze(0).unsqueeze(0)

class VectorialPolarPupil(Pupil):
    def __init__(self, e0x: float = 1.0, e0y: float = 0.0,
                 n_pix_pupil: int = 128, device: str = 'cpu',
                 zernike_coefficients: torch.Tensor | np.ndarray | list[float] | None = None):
        self.e0x = e0x
        self.e0y = e0y
        super().__init__(n_pix_pupil, device, zernike_coefficients)

    def initialize_field(self):
        single_field = torch.ones(self.n_pix_pupil).to(self.device)
        return torch.stack((self.e0x * single_field, self.e0y * single_field),
                            dim=0).to(torch.complex64).unsqueeze(0)

    def zernike_aberrations(self):
        aberrations = create_zernike_aberrations(self.zernike_coefficients, self.n_pix_pupil, mesh_type='polar')
        return aberrations.to(self.device).unsqueeze(0).unsqueeze(0)
