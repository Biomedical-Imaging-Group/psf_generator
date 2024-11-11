"""
Tests for pupils.

"""
import os

import numpy as np
import pytest

from src.psf_generator.propagators import *
from src.psf_generator.utils.handle_data import load_from_npy
from src.psf_generator.utils.misc import convert_tensor_to_array

N_PIX_PUPIL = 127

kwargs = {
            'n_pix_pupil': N_PIX_PUPIL,
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

@pytest.mark.parametrize('propagator_type, pupil_shape', [
    (ScalarCartesianPropagator, (N_PIX_PUPIL, N_PIX_PUPIL)),
    (ScalarSphericalPropagator, (N_PIX_PUPIL,)),
    (VectorialCartesianPropagator, (3, N_PIX_PUPIL, N_PIX_PUPIL)),
    (VectorialSphericalPropagator, (2, N_PIX_PUPIL)),
])
def test_pupil(propagator_type, pupil_shape):
    """
    Test if the pupil remains the same as the base one.

    This is useful when changes are made to the propagators.

    """
    is_vectorial = propagator_type in (VectorialCartesianPropagator, VectorialSphericalPropagator)
    if is_vectorial:
        kwargs.update({'e0x': 1.0, 'e0y': 0.0})

    propagator = propagator_type(**kwargs)
    pupil = propagator.get_pupil()
    assert tuple(pupil.shape) == pupil_shape

    filepath = os.path.join('results', 'data', f'{propagator.get_name()}_pupil_base.npy')
    # only save once and comment out for tests
    base_pupil = load_from_npy(filepath)
    # check content of PSFs
    np.testing.assert_allclose(base_pupil, convert_tensor_to_array(pupil))

