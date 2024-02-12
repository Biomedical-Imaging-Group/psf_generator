import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from pupil import ScalarInput
from params import Params
from utils.zoom_ifft2 import zoom_ifft2


class Propagator(ABC):
    def __init__(self, pupil, params: Params):
        self.pupil = pupil
        self.params = params

    @abstractmethod
    def compute_focus_field(self):
        pass

    @abstractmethod
    def display_psf(self):
        pass


class FourierPropagator(Propagator):
    """Simple Fourier propagation model (scalar)
        psf = |F^{-1} (pupil)|^2
    """
    def __init__(self, pupil, params):
        super().__init__(pupil, params)

        self.pupil = pupil
        self.params = params
        self.field = None

        if pupil is None:
            self.pupil = ScalarInput(None, params)

    def compute_focus_field(self):
        """compute the scalar field at focus from the scalar pupil function
        """
        pupil = self.pupil.return_pupil()
        self.field = torch.abs(zoom_ifft2(pupil, self.params)) ** 2

    def display_psf(self):
        if self.field is None:
            self.compute_focus_field()
        intensity = torch.abs(self.field) ** 2
        intensity = intensity / torch.max(intensity)
        plt.figure()
        plt.imshow(intensity)
        plt.show()
