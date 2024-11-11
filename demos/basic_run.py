"""
Example usage of all the propagators.

How to call them, visualize the results and save data.

"""
import os

from psf_generator.utils.handle_data import save_as_npy
from src.psf_generator.propagators import *
from src.psf_generator.utils.plots import plot_pupil, plot_psf

if __name__ == "__main__":
    n_pix_pupil = 127
    n_pix_psf = 256
    na = 1.4
    wavelength = 632
    fov = 2000
    defocus = 4000
    n_defocus = 200
    e0x = 1
    e0y = 1j
    mask = 'vortex'
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
    base_plot_path = os.path.join('results', 'plots', 'fields')
    os.makedirs(base_plot_path, exist_ok=True)
    base_data_path = os.path.join('results', 'data', 'fields')
    os.makedirs(base_data_path, exist_ok=True)

    # define propagators
    propagators = [
        ScalarCartesianPropagator(special_phase_mask=mask, **kwargs),
        ScalarSphericalPropagator(**kwargs),
        VectorialCartesianPropagator(e0x=e0x, e0y=e0y, special_phase_mask=mask, **kwargs),
        VectorialSphericalPropagator(e0x=e0x, e0y=e0y, **kwargs),
    ]

    for propagator in propagators:
        name = propagator.get_name()
        pupil = propagator.get_pupil()
        field = propagator.compute_focus_field()
        # save data as .npy
        for data, data_name in zip([pupil, field], ['pupil', 'psf']):
            filepath = os.path.join(base_data_path, f'{name}_{data_name}.npy')
            save_as_npy(filepath, data)
        # save parameters
        json_filepath = os.path.join(base_data_path, f'{name}_params.json')
        propagator.save_parameters(json_filepath)
        # plot results
        if 'cartesian' in name:
            filepath = os.path.join(base_plot_path, f'{name}_pupil.png')
            plot_pupil(pupil, name, filepath=filepath)

        quantities = ['modulus', 'phase', 'intensity']
        for quantity in quantities:
            filepath = os.path.join(base_plot_path, f'{name}_psf_{quantity}.png')
            plot_psf(field, name, quantity, filepath=filepath)
