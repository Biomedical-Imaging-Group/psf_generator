import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from params import Params
from utils.meshgrid import meshgrid_pupil


class FieldInput(ABC):
    def __init__(self, pupil_function, params: Params):
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

        self.params = params

        if pupil_function is None:
            self.create_pupil(self.params)

    def create_pupil(self, params: Params):
        """create a flat field on a disk as default pupil function
        """
        size = params.get_num('n_pix_pupil')

        kx, ky = meshgrid_pupil(params)
        kxy2 = kx ** 2 + ky ** 2
        disk_mask = kxy2 < params.get_phy('cut_off_freq') ** 2

        flat_field = torch.zeros(size, size, dtype=torch.complex64)

        self.pupil_function = torch.exp(1j * flat_field) * disk_mask

    def return_input(self):
        return self.pupil_function

    def display_input(self):
        plt.figure()
        plt.imshow(torch.real(self.pupil_function))
        plt.show()
