import torch

from params import Params
from utils.meshgrid import meshgrid_pupil


class ScalarPupil():
    def __init__(self, params: Params):
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


class VectorialPupil():
    def __init__(self, params: Params):
        self.params = params
        self.create_pupil(self.params)

    def create_pupil(self, params: Params):
        """create a vector field as default pupil function
        """
        size = params.get_num('n_pix_pupil')

        theta_max = torch.arcsin(params.get_phy('NA')*torch.ones(1)/params.get_phy('n_t'))
        theta_0 = torch.linspace(0, int(theta_max), size)
        phi_0 = torch.linspace(0, 2 * torch.pi, size)
        theta, phi = torch.meshgrid(theta_0, phi_0)
        f0 = params.get_num('filling_factor')
        E0 = 1 # to be added in params

        fw = torch.exp(- torch.sin(theta)**2 / (f0**2 * torch.sin(theta_max)**2))
        e_inc = E0 * fw

        e_x = e_inc/2 * ((1 + torch.cos(theta)) - (1 - torch.cos(theta)) * torch.cos(2 * phi)) * torch.sqrt(params.get_phy('n_t')*torch.ones(1)) * torch.sqrt(torch.cos(theta))
        e_y = e_inc/2 * (- (1 - torch.cos(theta)) * torch.sin(2 * phi)) * torch.sqrt(params.get_phy('n_t')*torch.ones(1)) * torch.sqrt(torch.cos(theta))
        e_z = - e_inc * torch.sin(theta) * torch.cos(phi) * torch.sqrt(params.get_phy('n_t')*torch.ones(1)) * torch.sqrt(torch.cos(theta))

        # self.pupil_function = torch.stack((e_x, e_y, e_z), dim=0)
        self.pupil_function = e_x

    def return_pupil(self):
        return self.pupil_function