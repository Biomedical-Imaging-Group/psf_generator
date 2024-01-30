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
        self._phy_params = {
            'psf_fov': 1000,
            'z_length': 500,
        }
        self._num_params = {
            'n_pix_pupil': 128,
            'n_pix_psf': 256,
            'n_pix_z': 128,
            'psf_zooming_factor': 0.025
        }

        # user input parameters
        self._phy_params['wavelength'] = user_input['wavelength'] \
            if user_input['wavelength'] is not None else 580
        self._phy_params['NA'] = user_input['NA'] \
            if user_input['NA'] is not None else 1.2
        self._phy_params['n_t'] = user_input['refractive_index'] \
            if user_input['refractive_index'] is not None else 1.1

        # computed parameters
        # 1. discretization parameters
        lamda = self._phy_params['wavelength']
        na = self._phy_params['NA']
        n_t = self._phy_params['n_t']
        psf_fov = self._phy_params['psf_fov']
        zooming = self._num_params['psf_zooming_factor']
        n_pix_pupil = self._num_params['n_pix_pupil']
        n_pix_psf = self._num_params['n_pix_psf']
        self._phy_params['cut_off_freq'] = 2 * torch.pi * na / lamda
        self._phy_params['max_freq'] = 2 * torch.pi * n_t / lamda
        self._phy_params['psf_pixel_size'] = psf_fov / n_pix_psf
        self._phy_params['pupil_fov'] = 2 * torch.pi * zooming / self._phy_params['psf_pixel_size']
        self._phy_params['pupil_pixel_size'] = self._phy_params['pupil_fov'] / n_pix_pupil
        self._num_params['pupil_radius'] = 0.4  # to remove
        # 2. czt parameters w and a
        czt_w = torch.exp(torch.tensor(2 * torch.pi * 1j * zooming / n_pix_pupil))
        czt_a = torch.exp(torch.tensor(torch.pi * 1j * zooming * n_pix_psf / n_pix_pupil))
        self._num_params['czt_w'] = czt_w
        self._num_params['czt_a'] = czt_a

        # GPU device
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        self._device = device

    def get_phy(self, key):
        return self._phy_params[key]

    def get_num(self, key):
        return self._num_params[key]

    @property
    def device(self):
        return self._device
