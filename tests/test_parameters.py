"""Tests for saving and restoring propagator parameters (to_dict / from_dict / save_parameters / load_parameters)."""
import json

import numpy as np
import pytest
import torch

from conftest import ALL_PROPAGATORS
from psf_generator.propagators import (
    PROPAGATORS,
    Propagator,
    ScalarCartesianPropagator,
    ScalarSphericalPropagator,
    VectorialCartesianPropagator,
    VectorialSphericalPropagator,
)
from psf_generator.utils.integrate import trapezoid_rule


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_from_dict_rebuilds_an_identical_propagator(make_propagator, propagator_type):
    original = make_propagator(propagator_type, zernike_coefficients=[0, 0, 0, 0, 0.3])
    parameters = original.to_dict()
    json.dumps(parameters)  # must be serialisable as is
    assert parameters['propagator'] == propagator_type.get_name()

    restored = Propagator.from_dict(parameters)
    assert type(restored) is propagator_type
    assert torch.equal(original.compute_focus_field(), restored.compute_focus_field())


def test_registry_lists_the_four_propagators():
    assert set(PROPAGATORS) == {'scalar_cartesian', 'scalar_spherical', 'vectorial_cartesian', 'vectorial_spherical'}
    for name, propagator_type in PROPAGATORS.items():
        assert propagator_type.get_name() == name


def test_save_and_load_parameters_round_trip(tmp_path):
    original = VectorialCartesianPropagator(
        n_pix_pupil=33, n_pix_psf=32, wavelength=500, na=1.2, pix_size=40, defocus_step=150, n_defocus=4,
        e0x=0.5 + 0.5j, e0y=-0.25j, zernike_coefficients=[0.0] * 12 + [0.4], special_phase_mask='vortex',
        sz_correction=False, apod_factor=True, envelope=0.9, gibson_lanni=True, n_s=1.33, z_p=2000,
        n_i=1.515, n_i0=1.518, device='cpu')
    path = tmp_path / 'nested' / 'params.json'
    original.save_parameters(str(path))

    with open(path) as file:
        saved = json.load(file)
    assert saved['e0x'] == [0.5, 0.5]
    assert saved['special_phase_mask'] == 'vortex'
    assert saved['sz_correction'] is False
    assert saved['n_i0'] == 1.518
    assert 'refractive_index' not in saved and 't_i' not in saved

    for loader in (VectorialCartesianPropagator.load_parameters, Propagator.load_parameters):
        restored = loader(str(path))
        assert type(restored) is VectorialCartesianPropagator
        assert restored.e0x == 0.5 + 0.5j and restored.e0y == -0.25j
        assert restored.special_phase_mask == 'vortex'
        assert restored.to_dict() == original.to_dict()
        assert torch.equal(restored.compute_focus_field(), original.compute_focus_field())


def test_spherical_integrator_is_stored_by_name(make_propagator):
    original = make_propagator(ScalarSphericalPropagator, integrator=trapezoid_rule, cos_factor=True)
    parameters = original.to_dict()
    assert parameters['integrator'] == 'trapezoid_rule'
    assert parameters['cos_factor'] is True
    restored = ScalarSphericalPropagator.from_dict(parameters)
    assert restored.integrator is trapezoid_rule
    with pytest.raises(ValueError):
        ScalarSphericalPropagator.from_dict({**parameters, 'integrator': 'gauss'})


def test_parameter_files_written_by_0_1_0_are_accepted():
    legacy = {
        'n_pix_pupil': 17, 'n_pix_psf': 16, 'device': 'cpu', 'zernike_coefficients': [0],
        'wavelength': 632, 'na': 1.3, 'pix_size': 20, 'refractive_index': 1.5, 'defocus_step': 0, 'n_defocus': 1,
        'apod_factor': False, 'envelope': None, 'gibson_lanni': False, 'z_p': 1000.0, 'n_s': 1.3, 'n_g': 1.5,
        'n_g0': 1.5, 't_g': 170000.0, 't_g0': 170000.0, 'n_i': 1.5, 't_i0': 100000.0, 't_i': 100000.0,
        'e0x': '(0.7071067811865476+0.7071067811865476j)', 'e0y': '1.0',
    }
    restored = VectorialSphericalPropagator.from_dict(legacy)
    assert restored.e0x == pytest.approx(0.7071067811865476 + 0.7071067811865476j)
    assert restored.e0y == 1.0
    assert restored.n_pix_psf == 16


def test_from_dict_validates_the_propagator_name(make_propagator):
    parameters = make_propagator(ScalarCartesianPropagator).to_dict()
    with pytest.raises(ValueError, match='not'):
        ScalarSphericalPropagator.from_dict(parameters)
    with pytest.raises(ValueError, match='Unknown propagator'):
        Propagator.from_dict({**parameters, 'propagator': 'kirchhoff'})
    with pytest.raises(ValueError, match="'propagator' key"):
        Propagator.from_dict({key: value for key, value in parameters.items() if key != 'propagator'})
    # a concrete class does not need the name
    assert isinstance(ScalarCartesianPropagator.from_dict(
        {key: value for key, value in parameters.items() if key != 'propagator'}), ScalarCartesianPropagator)


def test_tensor_masks_are_not_saved_and_warn(make_propagator):
    prop = make_propagator(ScalarCartesianPropagator, special_phase_mask=torch.zeros(63, 63))
    with pytest.warns(UserWarning, match='special_phase_mask'):
        parameters = prop.to_dict()
    assert parameters['special_phase_mask'] is None

    prop = make_propagator(VectorialSphericalPropagator, custom_field=torch.ones(63, dtype=torch.complex64))
    with pytest.warns(UserWarning, match='custom_field'):
        prop.to_dict()


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
@pytest.mark.parametrize('invalid, message', [
    ({'device': 'gpu'}, 'Invalid device'),
    ({'device': None}, 'Invalid device'),
    ({'n_pix_pupil': 1}, 'n_pix_pupil'),
    ({'n_pix_pupil': 0}, 'n_pix_pupil'),
    ({'n_pix_psf': 0}, 'n_pix_psf'),
    ({'n_defocus': 0}, 'n_defocus'),
    ({'wavelength': 0}, 'wavelength'),
    ({'wavelength': -632}, 'wavelength'),
    ({'pix_size': 0}, 'pix_size'),
    ({'pix_size': -10}, 'pix_size'),
    ({'na': 0}, 'na'),
    ({'na': -1.0}, 'na'),
    ({'na': 1.6, 'n_i0': 1.5}, 'numerical aperture'),
])
def test_invalid_parameters_are_rejected(make_propagator, propagator_type, invalid, message):
    with pytest.raises(ValueError, match=message):
        make_propagator(propagator_type, **invalid)


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_valid_edge_case_parameters_are_accepted(make_propagator, propagator_type):
    # na == n_i0 (a full hemisphere) and the smallest grids are allowed.
    make_propagator(propagator_type, na=1.5, n_i0=1.5)
    prop = make_propagator(propagator_type, n_pix_pupil=2, n_pix_psf=2, n_defocus=1)
    assert torch.isfinite(prop.compute_focus_field()).all()


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_numpy_scalars_are_accepted_as_sizes(make_propagator, propagator_type):
    """Sizes taken from an array (``np.int64`` is not a subclass of ``int``) must not be rejected."""
    prop = make_propagator(propagator_type, n_pix_pupil=np.int64(33), n_pix_psf=np.int64(32),
                           n_defocus=np.int64(2), wavelength=np.float32(500), pix_size=np.float64(50),
                           na=np.float32(1.2))
    assert torch.isfinite(prop.compute_focus_field()).all()


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_booleans_are_not_valid_sizes(make_propagator, propagator_type):
    with pytest.raises(ValueError, match='n_pix_psf'):
        make_propagator(propagator_type, n_pix_psf=True)
