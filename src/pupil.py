from abc import ABC, abstractmethod
import torch
from zernikepy import zernike_polynomials


class Pupil(ABC):
    def __init__(self, n_pix_pupil=128, device='cpu',
                 zernike_coefficients=(0,)):
        self.n_pix_pupil = n_pix_pupil
        self.device = device
        self.zernike_coefficients = zernike_coefficients
        self.field = None

    @abstractmethod
    def initialize_field(self):
        raise NotImplementedError


class ScalarCartesianPupil(Pupil):
    def __init__(self, n_pix_pupil=128, device='cpu', zernike_coefficients=(0,)):
        super().__init__(n_pix_pupil, device, zernike_coefficients)
        self.field = self.initialize_field()
        # self.field *= self.zernike_aberrations()

    def initialize_field(self):
        x = torch.linspace(-1, 1, self.n_pix_pupil).to(self.device)
        y = torch.linspace(-1, 1, self.n_pix_pupil).to(self.device)
        kx, ky = torch.meshgrid(x, y, indexing='xy')
        return (kx**2 + ky**2 < 1).to(torch.complex64).unsqueeze(0).unsqueeze(0)

    def zernike_aberrations(self):
        n_zernike = len(self.zernike_coefficients)
        zernike_basis = zernike_polynomials(mode=n_zernike-1, size=self.n_pix_pupil, select='all')
        zernike_phase = torch.sum(self.zernike_coefficients * torch.from_numpy(zernike_basis), dim=2)
        return torch.exp(1j * zernike_phase)


class ScalarPolarPupil(Pupil):
    def __init__(self, n_pix_pupil=128, device='cpu', zernike_coefficients=(0,)):
        super().__init__(n_pix_pupil, device, zernike_coefficients)
        self.field = self.initialize_field()

    def initialize_field(self):
        return torch.ones(self.n_pix_pupil).to(self.device).unsqueeze(0).unsqueeze(0)


class VectorialCartesianPupil(Pupil):
    def __init__(self, e0x=1, e0y=0, 
                 n_pix_pupil=128, device='cpu', zernike_coefficients=(0,)):
        super().__init__(n_pix_pupil, device, zernike_coefficients)
        self.e0x = e0x
        self.e0y = e0y

        self.field = self.initialize_field()
        # self.field *= self.zernike_aberrations()

    def initialize_field(self):
        x = torch.linspace(-1, 1, self.n_pix_pupil).to(self.device)
        y = torch.linspace(-1, 1, self.n_pix_pupil).to(self.device)
        kx, ky = torch.meshgrid(x, y, indexing='xy')
        single_field = (kx**2 + ky**2 < 1).to(torch.complex64).unsqueeze(0)
        return torch.stack((self.e0x * single_field, self.e0y * single_field), 
                           dim=0).unsqueeze(0)

    def zernike_aberrations(self):
        n_zernike = len(self.zernike_coefficients)
        zernike_basis = zernike_polynomials(mode=n_zernike-1, size=self.n_pix_pupil, select='all')
        zernike_phase = torch.sum(self.zernike_coefficients * torch.from_numpy(zernike_basis), dim=2)
        return torch.exp(1j * zernike_phase)


class VectorialPolarPupil(Pupil):
    def __init__(self, e0x=1, e0y=0, 
                 n_pix_pupil=128, device='cpu', zernike_coefficients=(0,)):
        super().__init__(n_pix_pupil, device, zernike_coefficients)
        self.e0x = e0x
        self.e0y = e0y
        self.field = self.initialize_field()

    def initialize_field(self):
        single_field = torch.ones(self.n_pix_pupil).to(self.device).unsqueeze(0)
        return torch.stack((self.e0x * single_field, self.e0y * single_field),
                            dim=0).unsqueeze(0)
