"""Physical-invariant tests for the computed PSFs.

Rather than comparing against stored baseline arrays (which are large and were
never committed), these tests assert properties that must hold for a correct,
aberration-free system:

* the output has the expected shape and is finite and non-negative in intensity;
* the computation is deterministic;
* the in-focus PSF peaks exactly at the grid centre and is symmetric;
* the Cartesian and spherical parameterisations, which model the same physics,
  agree on the normalized in-focus PSF.
"""
import pytest
import torch

from conftest import ALL_PROPAGATORS, VECTORIAL_PROPAGATORS
from psf_generator.propagators import (
    ScalarCartesianPropagator,
    ScalarSphericalPropagator,
    VectorialCartesianPropagator,
    VectorialSphericalPropagator,
)

N_PIX = 63
N_DEFOCUS = 3


def _intensity(field):
    """Sum ``|field|**2`` over the polarization axis -> ``(n_defocus, X, Y)``."""
    return (field.abs() ** 2).sum(dim=1)


def _in_focus_intensity(field):
    intensity = _intensity(field)
    return intensity[intensity.shape[0] // 2]


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_psf_shape(make_propagator, propagator_type):
    n_channels = 3 if propagator_type in VECTORIAL_PROPAGATORS else 1
    field = make_propagator(propagator_type).compute_focus_field()
    assert tuple(field.shape) == (N_DEFOCUS, n_channels, N_PIX, N_PIX)


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_psf_intensity_is_finite_and_non_negative(make_propagator, propagator_type):
    intensity = _intensity(make_propagator(propagator_type).compute_focus_field())
    assert torch.isfinite(intensity).all()
    assert (intensity >= 0).all()


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_psf_is_deterministic(make_propagator, propagator_type):
    first = make_propagator(propagator_type).compute_focus_field()
    second = make_propagator(propagator_type).compute_focus_field()
    assert torch.equal(first, second)


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_aberration_free_psf_peaks_at_centre(make_propagator, propagator_type):
    in_focus = _in_focus_intensity(make_propagator(propagator_type).compute_focus_field())
    n_y, n_x = in_focus.shape
    peak = divmod(int(torch.argmax(in_focus)), n_x)
    assert peak == (n_y // 2, n_x // 2)


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_aberration_free_psf_is_symmetric(make_propagator, propagator_type):
    in_focus = _in_focus_intensity(make_propagator(propagator_type).compute_focus_field())
    tolerance = 1e-4 * in_focus.max()
    assert (in_focus - torch.flip(in_focus, dims=[0])).abs().max() < tolerance
    assert (in_focus - torch.flip(in_focus, dims=[1])).abs().max() < tolerance


@pytest.mark.parametrize('cartesian_type, spherical_type', [
    (ScalarCartesianPropagator, ScalarSphericalPropagator),
    (VectorialCartesianPropagator, VectorialSphericalPropagator),
])
def test_cartesian_and_spherical_agree(make_propagator, cartesian_type, spherical_type):
    """The two parameterisations model the same physics (see the unifying
    framework in the paper) and must agree on the normalized in-focus PSF."""
    cartesian = _in_focus_intensity(make_propagator(cartesian_type).compute_focus_field())
    spherical = _in_focus_intensity(make_propagator(spherical_type).compute_focus_field())
    cartesian = cartesian / cartesian.max()
    spherical = spherical / spherical.max()
    assert (cartesian - spherical).abs().max() < 0.02
