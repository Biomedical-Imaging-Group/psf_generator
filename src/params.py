import torch


class Params:
    """
    Define parameters used internally in this library.
    - basic parameters: we provide a default value for every basic parameter.
      The user can overwrite the default values by specifying them from outside.
    - computed parameters: calculated from the basic parameters

    The parameters are defined in 2 dictionaries:
    - phy_params: physical or physic-related quantities, all the lengths in nm or nm^-1
    - num_params: numerical parameters, i.e. quantities in pixels
    """
    def __init__(self, user_input: dict):
        # basic parameters
        self._params = {'psf_fov': 1000,
                        'z_length': 500,
                        'n_pix_pupil': 128,
                        'n_pix_psf': 256,
                        'n_pix_z': 128,
                        'psf_zooming_factor': 0.025,
                        'number_of_zernike_modes': 15,
                        'initial_zernike_coeff_type': 'constant',
                        'initial_zernike_coeff_constant_value': 0.2,
                        'initial_coefficient_rand_factor': 0.5,
                        'wavelength': user_input['wavelength'] if user_input['wavelength'] is not None else 580,
                        'NA': user_input['NA'] if user_input['NA'] is not None else 1.2,
                        'n_t': user_input['refractive_index'] if user_input['refractive_index'] is not None else 1.1}

        # computed parameters
        # 1. discretization parameters
        lamda = self._params['wavelength']
        na = self._params['NA']
        n_t = self._params['n_t']
        psf_fov = self._params['psf_fov']
        zooming = self._params['psf_zooming_factor']
        n_pix_pupil = self._params['n_pix_pupil']
        n_pix_psf = self._params['n_pix_psf']
        self._params['cut_off_freq'] = 2 * torch.pi * na / lamda
        self._params['max_freq'] = 2 * torch.pi * n_t / lamda
        self._params['psf_pixel_size'] = psf_fov / n_pix_psf
        self._params['pupil_fov_phy'] = 2 * torch.pi * zooming / self._params['psf_pixel_size']
        self._params['pupil_pixel_size'] = self._params['pupil_fov_phy'] / n_pix_pupil
        self._params['pupil_radius_num'] = 0.4  # to remove
        self._params['filling_factor'] = torch.inf  # to be veified
        # 2. czt parameters w and a
        czt_w = torch.exp(torch.tensor(2 * torch.pi * 1j * zooming / n_pix_pupil))
        czt_a = torch.exp(torch.tensor(torch.pi * 1j * zooming * n_pix_psf / n_pix_pupil))
        self._params['czt_w'] = czt_w
        self._params['czt_a'] = czt_a
        # Zernike coefficients
        if self._params['initial_zernike_coeff_type'] == 'constant':
            self._params['zernike_coefficients'] = (torch.ones(self._params['number_of_zernike_modes'],
                                                                   dtype=torch.complex64) *
                                                        self._params['initial_zernike_coeff_constant_value'])
        elif self._params['initial_zernike_coeff_type'] == 'rand':
            self._params['zernike_coefficients'] = (torch.rand(self._params['number_of_zernike_modes'],
                                                                   dtype=torch.complex64) *
                                                        self._params['initial_coefficient_rand_factor'])

        else:
            self._params['zernike_coefficients'] = torch.zeros(self._params['number_of_zernike_modes'],
                                                                   dtype=torch.complex64)

        # GPU device
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        self._device = device

    def get(self, key):
        return self._params[key]

    @property
    def device(self):
        return self._device
