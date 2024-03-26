import torch

from zernikepy import zernike_polynomials

from params import Params
from utils.meshgrid import meshgrid_pupil


class ScalarPupil:
    def __init__(self, params: Params):
        self.pupil_function = None
        self.params = params
        self.create_pupil(self.params)

    def create_pupil(self, params: Params):
        """create a flat field on a disk as default pupil function
        """
        zernike_coefficients = params.get('zernike_coefficients')
        size = params.get('n_pix_pupil')

        kx, ky = meshgrid_pupil(params)
        kxy2 = kx ** 2 + ky ** 2
        pupil_amplitude = kxy2 < params.get('cut_off_freq') ** 2
        # creates a disk with radius cut_off_freq
        zernike_basis = zernike_polynomials(mode=params.get('number_of_zernike_modes')-1, size=size, select='all')
        pupil_phase = torch.sum(zernike_coefficients * torch.from_numpy(zernike_basis), dim=2)

        self.pupil_function = torch.exp(1j * pupil_phase) #* pupil_amplitude

    def return_pupil(self):
        return self.pupil_function


class VectorialPupil:
    def __init__(self, params: Params):
        self.pupil_function = None
        self.params = params
        self.create_pupil(self.params)

    def create_pupil(self, params: Params):
        """create a vector field as default pupil function
        """
        size = params.get('n_pix_pupil')
        theta_max = torch.asin(torch.tensor(params.get('NA')/params.get('n_t')))
        theta = torch.linspace(0, int(theta_max), size)
        a_x = 1  # to be inserted in parameters
        a_y = 0  # to be inserted in parameters
        f0 = 1  # to be inserted in parameters
        apod_funct = torch.exp(-torch.sin(theta)**2/f0**2/torch.sin(theta_max)**2)
        e_x = a_x * apod_funct
        e_y = a_y * apod_funct

        self.pupil_function = torch.stack((e_x, e_y), dim=0)

    def return_pupil(self):
        return self.pupil_function
