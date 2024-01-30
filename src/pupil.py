import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from params import Params


class FieldInput(ABC):
    def __init__(self, pupil_function, params):
        self.pupil_function = pupil_function
        self.params = params

    @abstractmethod
    def return_input(self):
        pass

    @abstractmethod
    def display_input(self):
        pass


class ScalarInput(FieldInput):
    def __init__(self, pupil_function, params: Params):
        super().__init__(pupil_function, params)

        self._n_pix = params.get_num('n_pix_pupil')
        self._pupil_radius = params.get_num('pupil_radius')

        if pupil_function is None:
            self.create_pupil(self._n_pix, self._pupil_radius)

    def create_pupil(self, n_pix, pupil_radius):
        x = torch.linspace(-1, 1, n_pix)
        y = torch.linspace(-1, 1, n_pix)
        xx, yy = torch.meshgrid(x, y)
        disk_mask = torch.sqrt(xx ** 2 + yy ** 2) < pupil_radius
        flat_field = torch.zeros(n_pix, n_pix, dtype=torch.complex64)
        self.pupil_function = torch.exp(1j * flat_field) * disk_mask

    def return_input(self):
        return self.pupil_function

    def display_input(self):
        plt.figure()
        plt.imshow(torch.real(self.pupil_function))
        plt.show()
