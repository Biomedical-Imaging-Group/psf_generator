from propagators import *
from utils.plots import plot_psf_intensity_maps


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
        'gibson_lanni': True
    }

    # define propagators
    propagators = [
        ScalarCartesianPropagator(**kwargs),
        ScalarSphericalPropagator(**kwargs),
        VectorialCartesianPropagator(e0x=e0x, e0y=e0y, **kwargs),
        VectorialSphericalPropagator(e0x=e0x, e0y=e0y, **kwargs)
    ]

    for propagator in propagators:
        class_name = propagator.__class__.__name__
        field = propagator.compute_focus_field()
        plot_psf_intensity_maps(field, class_name)

