import math
import os

from psf_generator.utils.handle_data import save_as_npy

from src.psf_generator.propagators import *

if __name__ == "__main__":
    n_pix_pupil = 127
    n_pix_psf = 256
    na = 1.3
    wavelength = 600
    fov = 2000
    defocus = 4000
    n_defocus = 256
    e0x = math.sqrt(2) / 2
    e0y = e0x * 1j
    mask = None
    zernike_coefficients = None

    kwargs = {
        'n_pix_pupil': n_pix_pupil,
        'n_pix_psf': n_pix_psf,
        'wavelength': wavelength,
        'zernike_coefficients': zernike_coefficients,
        'na': na,
        'fov': fov,
        'defocus_min': -defocus,
        'defocus_max': defocus,
        'n_defocus': n_defocus,
        'apod_factor': False,
        'gibson_lanni': False,
    }

    # define base file path
    exp_name = 'pure_gaussian_high_na_e0_1_1j'
    base_data_path = os.path.join('results', 'data', 'fields', exp_name)
    os.makedirs(base_data_path, exist_ok=True)

    # define propagators
    propagators = [
        ScalarCartesianPropagator(special_phase_mask=mask, **kwargs),
        VectorialCartesianPropagator(e0x=e0x, e0y=e0y, special_phase_mask=mask, **kwargs),
    ]

    for propagator in propagators:
        name = propagator.get_name()
        field = propagator.compute_focus_field()
        pupil = propagator.get_pupil()
        # save data as .npy
        for data, data_name in zip([field, pupil], ['psf', 'pupil']):
            filepath = os.path.join(base_data_path, f'{name}_{data_name}.npy')
            save_as_npy(filepath, data)
        # save parameters
        json_filepath = os.path.join(base_data_path, f'{name}_params.json')
        propagator.save_parameters(json_filepath)
