"""
Example usage of all the propagators.

How to call them and visualize the generated PSF.

"""
import os

import torch
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
    e0y = 0
    mask = 'vortex'
    zernike_coefficients = torch.ones(5)

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
        'gibson_lanni': True,
    }

    # define propagators
    propagators = [
        ScalarCartesianPropagator(special_phase_mask=mask, **kwargs),
        ScalarSphericalPropagator(**kwargs),
        VectorialCartesianPropagator(e0x=e0x, e0y=e0y, special_phase_mask=mask, **kwargs),
        VectorialSphericalPropagator(e0x=e0x, e0y=e0y, **kwargs),
    ]

    for propagator in propagators:
        name = propagator.get_name()
        pupil = propagator.get_input_field()
        field = propagator.compute_focus_field()
        base_path = os.path.join('results', 'plots')
        if 'cartesian' in name:
            filepath = os.path.join(base_path, f'{name}_pupil.png')
            plot_pupil(pupil, name, filepath=filepath)
        quantities = ['modulus', 'phase', 'intensity']
        for quantity in quantities:
            filepath = os.path.join(base_path, f'{name}_psf_{quantity}.png')
            plot_psf(field, name, quantity, filepath=filepath)
