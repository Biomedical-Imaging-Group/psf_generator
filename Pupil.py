import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class Pupil(ABC):
    def __init__(self, pupil_function, params):
        self.pupil_function = pupil_function
        self.params = params

    @abstractmethod
    def return_pupil(self):
        pass

    @abstractmethod
    def display_pupil(self):
        pass


class FourierPupil(Pupil):
    def __init__(self, pupil_function, params):
        super().__init__(pupil_function, params)

        self.n_pix = params.n_pix
        self.pupil_radius = params.pupil_radius

        if pupil_function is None:
            self.pupil_function = self.create_pupil(self.n_pix, self.pupil_radius)

    def create_pupil(self, n_pix, pupil_radius):
        x = torch.linspace(-1, 1, n_pix)
        y = torch.linspace(-1, 1, n_pix)
        xx, yy = torch.meshgrid(x, y)
        self.pupil_function = torch.sqrt(xx ** 2 + yy ** 2) < pupil_radius

    def return_pupil(self):
        return self.pupil_function

    def display_pupil(self):
        plt.figure()
        plt.imshow(self.pupil_function)
        plt.show()