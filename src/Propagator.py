import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from Pupil import ScalarInput


class Propagator(ABC):
    def __init__(self, input, params):
        self.input = input
        self.params = params

    @abstractmethod
    def compute_focus_field(self):
        pass

    @abstractmethod
    def display_psf(self):
        pass


class FourierPropagator(Propagator):
    def __init__(self, input, params):
        super().__init__(input, params)

        self.input = input
        self.n_pix = params.n_pix
        self.field = None

        if input is None:
            self.input = ScalarInput(None, params)

    def compute_focus_field(self):
        input = self.input.return_input()
        input = input.type(torch.complex64)
        input = torch.fft.fftshift(input)
        field = torch.fft.fft2(input)
        self.field = torch.fft.fftshift(field)

    def display_psf(self):
        if self.field is None:
            self.compute_focus_field()
        intensity = torch.abs(self.field)**2
        intensity = intensity / torch.max(intensity)
        plt.figure()
        plt.imshow(intensity)
        plt.show()
