import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from params import Params
from utils.meshgrid import meshgrid_pupil


class FieldPupil(ABC):
    def __init__(self, pupil_function, params: Params):
        self.pupil_function = pupil_function
        self.params = params

    @abstractmethod
    def return_pupil(self):
        pass

    @abstractmethod
    def display_pupil(self):
        pass


class ScalarPupil(FieldPupil):
    def __init__(self, params: Params):
        super().__init__(params)

        self.params = params
        self.create_pupil(self.params)

    def create_pupil(self, params: Params):
        """create a flat field on a disk as default pupil function
        """
        size = params.get_num('n_pix_pupil')

        kx, ky = meshgrid_pupil(params)
        kxy2 = kx ** 2 + ky ** 2
        pupil_amplitude = kxy2 < params.get_phy('cut_off_freq') ** 2
        # creates a disk with radius cut_off_freq

        pupil_phase = torch.zeros(size, size, dtype=torch.complex64)

        self.pupil_function = torch.exp(1j * pupil_phase) * pupil_amplitude

    def return_pupil(self):
        return self.pupil_function

    def display_pupil(self):
        plt.figure()
        plt.imshow(torch.real(self.pupil_function))
        plt.show()
