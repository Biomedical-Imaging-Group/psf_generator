"""
Example usage of all the propagators.

How to call them and visualize the generated PSF.

"""
import os

from src.psf_generator.propagators import *
from src.psf_generator.utils.plots import plot_psf_intensity_maps

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

    kwargs = {
        'n_pix_pupil': n_pix_pupil,
        'n_pix_psf': n_pix_psf,
        'wavelength': wavelength,
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
        ScalarCartesianPropagator(**kwargs),
        ScalarSphericalPropagator(**kwargs),
        VectorialCartesianPropagator(e0x=e0x, e0y=e0y, **kwargs),
        VectorialSphericalPropagator(e0x=e0x, e0y=e0y, **kwargs),
    ]

    for propagator in propagators:
        name = propagator.get_name()
        field = propagator.compute_focus_field()
        filepath = os.path.join('results', 'plots', f'{name}_psf.png')
        plot_psf_intensity_maps(field, name, filepath=filepath)

