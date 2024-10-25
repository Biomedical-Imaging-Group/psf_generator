"""
Tests for PSFs.

"""
import os

import numpy as np
import pytest

from src.psf_generator.propagators import *
from src.psf_generator.utils.handle_data import load_from_npy
from src.psf_generator.utils.misc import convert_tensor_to_array

kwargs = {
            'n_pix_pupil': 127,
            'n_pix_psf': 256,
            'wavelength': 632,
            'na': 1.4,
            'fov': 2000,
            'defocus_min': -1000,
            'defocus_max': 1000,
            'n_defocus': 126,
            'apod_factor': False,
            'gibson_lanni': False,
        }

@pytest.mark.parametrize('propagator_type', [
    ScalarCartesianPropagator,
    ScalarSphericalPropagator,
    VectorialCartesianPropagator,
    VectorialSphericalPropagator,
])
def test_psf(propagator_type):
    """
    Test if the PSF remains the same as the base one.

    This is useful when changes are made to the propagators.

    """
    is_vectorial = propagator_type in (VectorialCartesianPropagator, VectorialSphericalPropagator)
    if is_vectorial:
        kwargs.update({'e0x': 1.0, 'e0y': 0.0})

    propagator = propagator_type(**kwargs)
    n_channels = 3 if is_vectorial else 1
    psf = propagator.compute_focus_field()
    # check size of psf
    assert psf.shape == (kwargs['n_defocus'], n_channels, kwargs['n_pix_psf'], kwargs['n_pix_psf'])

    filepath = os.path.join('results', 'data', f'{propagator.get_name()}_psf_base.npy')
    base_psf = load_from_npy(filepath)
    # check content of PSFs
    np.testing.assert_allclose(base_psf, convert_tensor_to_array(psf))

