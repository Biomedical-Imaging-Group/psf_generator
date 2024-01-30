import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


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
    def __init__(self, pupil_function, params):
        super().__init__(pupil_function, params)

        self.n_pix = params.n_pix
        self.pupil_radius = params.pupil_radius

        if pupil_function is None:
            self.create_input(self.n_pix, self.pupil_radius)

    def create_input(self, n_pix, pupil_radius):
        x = torch.linspace(-1, 1, n_pix)
        y = torch.linspace(-1, 1, n_pix)
        xx, yy = torch.meshgrid(x, y)
        self.pupil_function = (torch.sqrt(xx ** 2 + yy ** 2) < pupil_radius).type(torch.complex64)

    def return_input(self):
        return self.pupil_function

    def display_input(self):
        plt.figure()
        plt.imshow(torch.real(self.pupil_function))
        plt.show()
