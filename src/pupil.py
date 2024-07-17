import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import binom
from zernikepy import zernike_polynomials


class Pupil(ABC):
    def __init__(self, n_pix_pupil=128, device='cpu',
                 zernike_coefficients=[0,]):
        self.n_pix_pupil = n_pix_pupil
        self.device = device
        self.zernike_coefficients = zernike_coefficients
        self.field = None

    @abstractmethod
    def initialize_field(self):
        raise NotImplementedError


class ScalarCartesianPupil(Pupil):
    '''
    Define a 2D pupil function for the scalar Cartesian case. The function is defined on the 
    unit disk centered at (0,0): u ** 2 + v ** 2 <= 1. The mapping between this domain and
    the physical pupil coordinates are:

        u = sx / s_max
        v = sy / s_max

    such that the physical domain is:

        sx ** 2 + sy ** 2 <= s_max ** 2 = sin(theta_max) ** 2
    '''
    def __init__(self, n_pix_pupil=128, device='cpu', zernike_coefficients=[0,]):
        super().__init__(n_pix_pupil, device, zernike_coefficients)
        self.field = self.initialize_field()
        self.field *= self.zernike_aberrations()

    def initialize_field(self):
        x = torch.linspace(-1, 1, self.n_pix_pupil)
        y = torch.linspace(-1, 1, self.n_pix_pupil)
        kx, ky = torch.meshgrid(x, y, indexing='xy')
        return (kx**2 + ky**2 <= 1).to(torch.complex64).unsqueeze(0).unsqueeze(0).to(self.device)

    def zernike_aberrations(self):
        n_zernike = len(self.zernike_coefficients)
        zernike_basis = zernike_polynomials(mode=n_zernike-1, size=self.n_pix_pupil, select='all')
        zernike_coefficients = torch.tensor(self.zernike_coefficients).reshape(1, 1, n_zernike)
        zernike_phase = torch.sum(zernike_coefficients * zernike_basis, dim=2)
        return torch.exp(1j * zernike_phase).to(torch.complex64).to(self.device).unsqueeze(0).unsqueeze(0)


class ScalarPolarPupil(Pupil):
    '''
    Define a (1D) radial pupil function for the scalar polar case. The function is defined on 
    the interval `\rho` \in [0,1]; `\rho` is a "normalized" radius. The conversion to physical
    pupil coordinates - the polar angle `\theta` - is given by:

        \rho = \frac{\sin{\theta}}{\sin{\theta_{max}}}

    such that the physical domain is:

        \theta \leq \theta_{max}
    '''
    def __init__(self, n_pix_pupil=128, device='cpu', zernike_coefficients=[0,]):
        super().__init__(n_pix_pupil, device, zernike_coefficients)
        self.field = self.initialize_field()
        self.field *= self.zernike_aberrations()

    def initialize_field(self):
        return torch.ones(self.n_pix_pupil).to(torch.complex64).to(self.device).unsqueeze(0).unsqueeze(0)

    def zernike_aberrations(self):
        n_zernike = len(self.zernike_coefficients)
        rho = torch.linspace(0, 1, self.n_pix_pupil)
        phi = 0
        zernike_phase = torch.zeros(self.n_pix_pupil)
        for i in range(n_zernike):
            n, l = self.index_to_nl(i)
            curr_coef = self.zernike_coefficients[i]
            if l != 0 and curr_coef != 0:
                print("Warning: Zernike coefficients for l != 0 are not supported in polar coordinates.")
            elif l == 0:
                zernike_phase += curr_coef * torch.tensor(self._zernike_nl(n, l, rho, phi))
        return torch.exp(1j * zernike_phase).to(self.device).unsqueeze(0).unsqueeze(0)

    def eval_field_at(self, r):
        '''
        Evaluate the pupil field at the radius `rho` = `r`.
        '''
        n_zernike = len(self.zernike_coefficients)
        zernike_phase = torch.zeros_like(r)
        for i in range(n_zernike):
            n, l = self.index_to_nl(i)
            curr_coef = self.zernike_coefficients[i]
            if l != 0 and curr_coef != 0:
                print("Warning: Zernike coefficients for l != 0 are not supported in polar coordinates.")
            elif l == 0:
                zernike_phase += curr_coef * torch.tensor(self._zernike_nl(n, l, rho=r, phi=0.0))
        return torch.exp(1j * zernike_phase).to(self.device).unsqueeze(0).unsqueeze(0)

    def eval_field_at_np(self, r):
        '''
        Evaluate the pupil field at the radius `rho` = `r`. This version is implemented in numpy and is
        used to generate the ground truth PSF field.
        '''
        n_zernike = len(self.zernike_coefficients)
        zernike_phase = np.zeros_like(r)
        for i in range(n_zernike):
            n, l = self.index_to_nl(i)
            curr_coef = self.zernike_coefficients[i]
            if l != 0 and curr_coef != 0:
                print("Warning: Zernike coefficients for l != 0 are not supported in polar coordinates.")
            elif l == 0:
                zernike_phase += curr_coef * self._zernike_nl(n, l, rho=r, phi=0.0)
        return np.exp(1j * zernike_phase)

    @staticmethod
    def index_to_nl(index):
        n = 0
        while True:
            for l in range(n + 1):
                if n*(n+1)/2 + l == index:
                    return (n, -n+2*l)
                elif n*(n+1)/2 + l > index:
                    raise ValueError('Index out of bounds.')
            n += 1

    @staticmethod
    def _zernike_nl(n: int, l: int, rho: float, phi: float):
        m = abs(l)
        R = 0
        for k in np.arange(0, (n - m) / 2 + 1):
            R = R + (-1) ** k * binom(n - k, k) * binom(n - 2 * k, (n - m) / 2 - k) * rho ** (n - 2 * k)
        Z = np.where(rho <= 1, R, 0)
        Z *= np.cos(m * phi) if l >= 0 else np.sin(m * phi)
        return Z

class VectorialCartesianPupil(Pupil):
    def __init__(self, e0x=1, e0y=0, 
                 n_pix_pupil=128, device='cpu', zernike_coefficients=(0,)):
        super().__init__(n_pix_pupil, device, zernike_coefficients)
        self.e0x = e0x
        self.e0y = e0y

        self.field = self.initialize_field()
        self.field *= self.zernike_aberrations()

    def initialize_field(self):
        x = torch.linspace(-1, 1, self.n_pix_pupil)
        y = torch.linspace(-1, 1, self.n_pix_pupil)
        kx, ky = torch.meshgrid(x, y, indexing='xy')
        single_field = (kx**2 + ky**2 < 1).to(torch.complex64)
        return torch.stack((self.e0x * single_field, self.e0y * single_field), 
                           dim=0).unsqueeze(0).to(self.device)

    def zernike_aberrations(self):
        n_zernike = len(self.zernike_coefficients)
        zernike_basis = zernike_polynomials(mode=n_zernike-1, size=self.n_pix_pupil, select='all')
        zernike_coefficients = torch.tensor(self.zernike_coefficients).reshape(1, 1, n_zernike)
        zernike_phase = torch.sum(zernike_coefficients * torch.from_numpy(zernike_basis), dim=2)
        return torch.exp(1j * zernike_phase).to(torch.complex64).to(self.device).unsqueeze(0).unsqueeze(0)


class VectorialPolarPupil(Pupil):
    def __init__(self, e0x=1, e0y=0, 
                 n_pix_pupil=128, device='cpu', zernike_coefficients=(0,)):
        super().__init__(n_pix_pupil, device, zernike_coefficients)
        self.e0x = e0x
        self.e0y = e0y
        self.field = self.initialize_field()
        self.field *= self.zernike_aberrations()

    def initialize_field(self):
        single_field = torch.ones(self.n_pix_pupil).to(self.device)
        return torch.stack((self.e0x * single_field, self.e0y * single_field),
                            dim=0).to(torch.complex64).unsqueeze(0)

    def zernike_aberrations(self):
        n_zernike = len(self.zernike_coefficients)
        rho = torch.sin(torch.linspace(0, 1, self.n_pix_pupil))
        phi = 0
        zernike_phase = torch.zeros(self.n_pix_pupil)
        for i in range(n_zernike):
            n, l = self.index_to_nl(i)
            curr_coef = self.zernike_coefficients[i]
            if l != 0 and curr_coef != 0:
                print("Warning: Zernike coefficients for l != 0 are not supported in polar coordinates.")
            elif l == 0:
                zernike_phase += curr_coef * torch.tensor(self._zernike_nl(n, l, rho, phi))
        return torch.exp(1j * zernike_phase).to(torch.complex64).to(self.device).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def index_to_nl(index):
        n = 0
        while True:
            for l in range(n + 1):
                if n*(n+1)/2 + l == index:
                    return (n, -n+2*l)
                elif n*(n+1)/2 + l > index:
                    raise ValueError('Index out of bounds.')
            n += 1

    @staticmethod
    def _zernike_nl(n: int, l: int, rho: float, phi: float):
        m = abs(l)
        R = 0
        for k in np.arange(0, (n - m) / 2 + 1):
            R = R + (-1) ** k * binom(n - k, k) * binom(n - 2 * k, (n - m) / 2 - k) * rho ** (n - 2 * k)
        Z = np.where(rho <= 1, R, 0)
        Z *= np.cos(m * phi) if l >= 0 else np.sin(m * phi)
        return Z
