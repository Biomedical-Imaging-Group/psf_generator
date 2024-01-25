import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from Pupil import FourierPupil

class Propagator(ABC):
    def __init__(self, pupil, params):
        self.pupil = pupil
        self.params = params

    @abstractmethod
    def compute_focus_field(self):
        pass

    @abstractmethod
    def display_field(self):
        pass

class FourierPropagator(Propagator):
    def __init__(self, pupil, params):
        super().__init__(pupil, params)

        self.n_pix = params.n_pix
        self.field = None

        if pupil is None:
            self.pupil = FourierPupil(None, params)

    def compute_focus_field(self):
        pupil = self.pupil.return_pupil()
        pupil = pupil.type(torch.complex64)
        pupil = torch.fft.fftshift(pupil)
        field = torch.fft.fft2(pupil)
        self.field = torch.fft.fftshift(field)

    def display_field(self):
        if self.field is None:
            self.compute_focus_field()
        intensity = torch.abs(self.field)**2
        intensity = intensity / torch.max(intensity)
        plt.figure()
        plt.imshow(intensity)
        plt.show()