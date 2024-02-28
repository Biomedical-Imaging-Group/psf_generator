import torch
from abc import ABC, abstractmethod
from pupil import ScalarPupil,VectorialPupil
from params import Params
from utils.custom_ifft2 import custom_ifft2
from utils.integrate import integrate_summation_rule, integrate_double_summation_rule
from scipy.special import jv

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


class SimpleVectorial(Propagator):
    """Richards-Wolf model (vectorial) for x-polarized plane wave incident on
    the lens. Here I keep only the first term of the x-component of the electric field.
    """

    def __init__(self, pupil, params):
        super().__init__(pupil, params)

        self.pupil = pupil if pupil else VectorialPupil(params)
        self.params = params
        self.field = None

    def compute_focus_field(self):
        """Compute the vectorial field at focus.
        Here, it doesn't take as input the pupil function.
        """
        size = self.params.get_num('n_pix_pupil')
        x = torch.linspace(-2 * self.params.get_phy('wavelength'), 2 * self.params.get_phy('wavelength'), size)
        y = torch.linspace(-2 * self.params.get_phy('wavelength'), 2 * self.params.get_phy('wavelength'), size)
        xx, yy = torch.meshgrid(x, y)
        zz = torch.zeros(1)
        theta_max = torch.asin(self.params.get_phy('NA') * torch.ones(1) / self.params.get_phy('n_t'))
        i0 = integrate_summation_rule(lambda theta: self.integrand00(theta, xx, yy, zz), 0, theta_max, size)
        i2 = integrate_summation_rule(lambda theta: self.integrand02(theta, xx, yy, zz), 0, theta_max, size)
        i1 = integrate_summation_rule(lambda theta: self.integrand01(theta, xx, yy, zz), 0, theta_max, size)
        varphi = torch.atan2(yy, xx)
        field_x = i0 + i2*torch.cos(2*varphi)
        field_y = i2 * torch.sin(2 * varphi)
        field_z = -2 * 1j * i1 * torch.cos(varphi)
        self.field = torch.stack((field_x, field_y, field_z), dim=0)

        return torch.abs(self.field) ** 2

    def integrand00(self, theta, xx, yy, zz):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get_phy('n_t') / self.params.get_phy('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j0 = jv(0, r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t * (1 + cos_t)
        return torch.multiply(i, j0)

    def integrand02(self, theta, xx, yy, zz):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get_phy('n_t') / self.params.get_phy('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j2 = jv(2, r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t * (1 - cos_t)
        return torch.multiply(i, j2)

    def integrand01(self, theta, xx, yy, zz):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get_phy('n_t') / self.params.get_phy('wavelength')
        r = k * torch.sqrt(xx ** 2 + yy ** 2) * sin_t
        j1 = jv(1, r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t ** 2
        return torch.multiply(i, j1)


class ComplexVectorial(Propagator):

    def __init__(self, pupil, params):
        super().__init__(pupil, params)

        self.pupil = pupil if pupil else VectorialPupil(params)
        self.params = params
        self.field = None

    def compute_focus_field(self):
        """Compute the vectorial field at focus.
        Here, it doesn't take as input the pupil function.
        """
        size = self.params.get_num('n_pix_pupil')

        z = torch.ones(1)
        p = torch.zeros(3)
        x = torch.linspace(-2 * self.params.get_phy('wavelength'), 2 * self.params.get_phy('wavelength'), size)
        y = torch.linspace(-2 * self.params.get_phy('wavelength'), 2 * self.params.get_phy('wavelength'), size)
        zz, pp, xx, yy = torch.meshgrid(z, p, x, y)

        theta_max = torch.asin(torch.tensor(self.params.get_phy('NA') / self.params.get_phy('n_t')))
        self.field = integrate_double_summation_rule(lambda theta, phi: self.integrand2(theta, phi, zz, xx, yy),
                                                     0, theta_max, 0, 2 * torch.pi, size)

        return torch.abs(self.field).squeeze() ** 2

    def integrand2(self, theta, phi, zz, xx, yy):
        k = 2 * torch.pi * self.params.get_phy('n_t') / self.params.get_phy('wavelength')
        r = torch.sqrt(xx ** 2 + yy ** 2)
        psi = torch.atan2(yy, xx)

        e = self.pupil.create_pupil_function(theta, phi, self.params)
        e = e.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        i = e * torch.exp(1j * k * zz * torch.cos(theta)) * torch.exp(1j * k * r * torch.sin(theta) *
                                                                      torch.cos(phi - psi)) * torch.sin(theta)
        return i
