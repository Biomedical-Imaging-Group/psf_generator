import torch
from abc import ABC, abstractmethod
from pupil import ScalarPupil, VectorialPupil
from params import Params
from utils.custom_ifft2 import custom_ifft2
from utils.integrate import integrate_summation_rule
from torch import special as sp


class Propagator(ABC):
    def __init__(self, pupil, params: Params):
        self.pupil = pupil
        self.params = params
        self.field = None

    @abstractmethod
    def compute_focus_field(self):
        raise NotImplementedError

    def get_focus_field(self):
        if self.field is None:
            self.field = self.compute_focus_field()
        return self.field


class FourierPropagator(Propagator):
    """Simple Fourier propagation model (scaler)
        psf = |F^{-1} (pupil)|^2
    """

    def __init__(self, pupil, params):
        super().__init__(pupil, params)

        self.pupil = pupil
        self.params = params
        self.field = None

        if pupil is None:
            self.pupil = ScalarPupil(params)

    def compute_focus_field(self):
        """compute the scaler field at focus from the scaler pupil function
        """
        pupil = self.pupil.return_pupil()
        self.field = torch.abs(custom_ifft2(pupil, self.params)) ** 2
        return self.field

class KirchhoffPropagator(Propagator):
    """Kirchhoff model (scaler). """

    def __init__(self, pupil, params):
        super().__init__(pupil, params)

        self.pupil = pupil
        self.params = params
        self.field = None

        if pupil is None:
            self.pupil = ScalarPupil(params)

    def compute_focus_field(self):
        """Compute the scaler field at focus.
        """

        pupil = self.pupil.return_pupil()
        size = self.params.get('n_pix_pupil')
        x = torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        y = torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        z = torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        xx, yy, zz = torch.meshgrid(x, y, z,  indexing='ij')
        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        self.field = integrate_summation_rule(lambda theta: self.integrand00(theta, xx, yy, zz, pupil), 0, theta_max, size)
        return torch.abs(self.field) ** 2

    def integrand00(self, theta, xx, yy, zz, pupil):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j0 = sp.bessel_j0(r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t

        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]

        return torch.multiply(i, j0)

class Vectorial(Propagator):
    """Richards-Wolf model (vectorial). """

    def __init__(self, pupil, params):
        super().__init__(pupil, params)

        self.pupil = pupil
        self.params = params
        self.field = None

        if pupil is None:
            self.pupil = VectorialPupil(params)

    def compute_focus_field(self):
        """Compute the vectorial field at focus.
        """

        pupil = self.pupil.return_pupil()
        size = self.params.get('n_pix_pupil')
        x = torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        y = torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        z =  torch.linspace(-2 * self.params.get('wavelength'), 2 * self.params.get('wavelength'), size)
        xx, yy, zz  = torch.meshgrid(x, y, z,  indexing='ij')
        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))

        i0_x = integrate_summation_rule(lambda theta: self.integrand00(theta, xx, yy, zz, pupil[0]), 0, theta_max, size)
        i2_x = integrate_summation_rule(lambda theta: self.integrand02(theta, xx, yy, zz, pupil[0]), 0, theta_max, size)
        i1_x = integrate_summation_rule(lambda theta: self.integrand01(theta, xx, yy, zz, pupil[0]), 0, theta_max, size)

        i0_y = integrate_summation_rule(lambda theta: self.integrand00(theta, xx, yy, zz, pupil[1]), 0, theta_max, size)
        i2_y = integrate_summation_rule(lambda theta: self.integrand02(theta, xx, yy, zz, pupil[1]), 0, theta_max, size)
        i1_y = integrate_summation_rule(lambda theta: self.integrand01(theta, xx, yy, zz, pupil[1]), 0, theta_max, size)

        varphi = torch.atan2(yy, xx)
        field_x = i0_x + i2_x * torch.cos(2 * varphi) + i2_y * torch.sin(2 * varphi)
        field_y = i2_x * torch.sin(2 * varphi) + i0_y - i2_y * torch.cos(2 * varphi)
        field_z = -2 * 1j * i1_x * torch.cos(varphi) - 2 * 1j * i1_y * torch.sin(varphi)

        self.field = torch.stack((field_x, field_y, field_z), dim=0).movedim(-1,0)

        return torch.abs(self.field) ** 2

    def integrand00(self, theta, xx, yy, zz, pupil):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j0 = sp.bessel_j0(r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t * (1 + cos_t)

        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]

        return torch.multiply(i, j0)

    def integrand02(self, theta, xx, yy, zz, pupil):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        eps = 1e-10
        j2 = 2 * sp.bessel_j1(r) / (r+eps) - sp.bessel_j0(r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t * (1 - cos_t)

        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]

        return torch.multiply(i, j2)

    def integrand01(self, theta, xx, yy, zz, pupil):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get('n_t') / self.params.get('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j1 = sp.bessel_j1(r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t ** 2

        theta_max = torch.asin(self.params.get('NA') * torch.ones(1) / self.params.get('n_t'))
        theta_0 = theta_max/self.params.get('n_pix_pupil')
        i *= pupil[int(theta/theta_0)]

        return torch.multiply(i, j1)

