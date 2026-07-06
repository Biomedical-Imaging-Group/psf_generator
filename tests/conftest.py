"""Shared fixtures and configuration for the test suite."""
import pytest

from psf_generator.propagators import (
    ScalarCartesianPropagator,
    ScalarSphericalPropagator,
    VectorialCartesianPropagator,
    VectorialSphericalPropagator,
)

# A small, fast configuration used across the suite.
#
# The pixel counts are odd on purpose:
#   * an odd ``n_pix_psf`` gives an unambiguous grid centre at ``n // 2``, so the
#     aberration-free PSF peaks exactly at the centre for every propagator;
#   * an odd ``n_pix_pupil`` gives the spherical integrator its high-order
#     accuracy (an even pupil size triggers a low-accuracy warning).
SMALL_KWARGS = dict(
    n_pix_pupil=63,
    n_pix_psf=63,
    wavelength=632,
    na=1.4,
    pix_size=100,
    defocus_step=200,
    n_defocus=3,
    apod_factor=False,
    gibson_lanni=False,
)

SCALAR_PROPAGATORS = [ScalarCartesianPropagator, ScalarSphericalPropagator]
VECTORIAL_PROPAGATORS = [VectorialCartesianPropagator, VectorialSphericalPropagator]
ALL_PROPAGATORS = SCALAR_PROPAGATORS + VECTORIAL_PROPAGATORS


@pytest.fixture
def make_propagator():
    """Return a factory that builds a propagator with the shared small config.

    Vectorial propagators additionally receive a linearly x-polarized input
    field. Any keyword can be overridden per call.
    """
    def _make(propagator_type, **overrides):
        kwargs = dict(SMALL_KWARGS)
        if propagator_type in VECTORIAL_PROPAGATORS:
            kwargs.update(e0x=1.0, e0y=0.0)
        kwargs.update(overrides)
        return propagator_type(**kwargs)

    return _make
