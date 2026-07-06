"""Tests for the pupil functions of each propagator.

These check structural and determinism properties that must hold whenever the
propagator internals change, without relying on external baseline files.
"""
import pytest
import torch

from conftest import ALL_PROPAGATORS

N_PIX = 63

# Actual shapes returned by ``get_pupil`` for the shared small config.
EXPECTED_PUPIL_SHAPES = {
    'scalar_cartesian': (1, 1, N_PIX, N_PIX),
    'scalar_spherical': (N_PIX,),
    'vectorial_cartesian': (1, 3, N_PIX, N_PIX),
    'vectorial_spherical': (2, N_PIX),
}


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_pupil_shape(make_propagator, propagator_type):
    prop = make_propagator(propagator_type)
    pupil = prop.get_pupil()
    assert tuple(pupil.shape) == EXPECTED_PUPIL_SHAPES[prop.get_name()]


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_pupil_is_finite_and_nonzero(make_propagator, propagator_type):
    pupil = make_propagator(propagator_type).get_pupil()
    assert torch.isfinite(pupil).all()
    assert pupil.abs().sum() > 0


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_pupil_is_deterministic(make_propagator, propagator_type):
    first = make_propagator(propagator_type).get_pupil()
    second = make_propagator(propagator_type).get_pupil()
    assert torch.equal(first, second)
