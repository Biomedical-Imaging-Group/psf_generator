import torch
from abc import ABC, abstractmethod
from pupil import ScalarPupil
from params import Params
from utils.custom_ifft2 import custom_ifft2
from utils.integrate import integrate_summation_rule
from scipy.special import jv

class Propagator(ABC):
    def __init__(self, pupil, params: Params):
        self.pupil = pupil
        self.params = params

    @abstractmethod
    def compute_focus_field(self):
        pass

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


class SimpleVectorial(Propagator):
    """Richards-Wolf model (vectorial) for x-polarized plane wave incident on
    the lens. Here I keep only the first term of the x-component of the electric field.
    """
    def __init__(self, pupil, params):
        super().__init__(pupil, params)

        self.pupil = pupil
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
        self.field = integrate_summation_rule(lambda theta: self.integrand(theta, xx, yy, zz), 0, theta_max, size)
        return self.field

    def integrand(self, theta, xx, yy, zz):
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        k = 2 * torch.pi * self.params.get_phy('n_t') / self.params.get_phy('wavelength')
        r = k * torch.sqrt(xx**2 + yy**2) * sin_t
        j0 = jv(0, r)

        # make sure i is complex
        i = torch.exp(1j * k * zz * cos_t)
        i *= torch.sqrt(cos_t) * sin_t * (1 + cos_t)
        return torch.multiply(i, j0)

