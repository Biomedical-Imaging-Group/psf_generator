"""Physical-invariant tests for the computed PSFs.

Rather than comparing against stored baseline arrays (which are large and were
never committed), these tests assert properties that must hold for a correct,
aberration-free system:

* the output has the expected shape and is finite and non-negative in intensity;
* the computation is deterministic;
* the in-focus PSF peaks exactly at the grid centre and is symmetric;
* the Cartesian and spherical parameterisations, which model the same physics,
  agree on the normalized in-focus PSF;
* the grid is physical: z-slices are exactly ``defocus_step`` apart, pixels are
  exactly ``pix_size`` apart, and index ``n // 2`` is the optical axis / focal
  plane along every axis.
"""
import math

import pytest
import torch
from torch.special import bessel_j1

from conftest import ALL_PROPAGATORS, VECTORIAL_PROPAGATORS
from psf_generator.propagators import (
    ScalarCartesianPropagator,
    ScalarSphericalPropagator,
    VectorialCartesianPropagator,
    VectorialSphericalPropagator,
)
from psf_generator.utils.zernike import create_zernike_aberrations, zernike_basis

N_PIX = 63
N_DEFOCUS = 3


def _intensity(field):
    """Sum ``|field|**2`` over the polarization axis -> ``(n_defocus, X, Y)``."""
    return (field.abs() ** 2).sum(dim=1)


def _in_focus_intensity(field):
    intensity = _intensity(field)
    return intensity[intensity.shape[0] // 2]


def _normalized_in_focus_intensity(field):
    in_focus = _in_focus_intensity(field)
    return in_focus / in_focus.max()


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


PARAMETERISATION_PAIRS = [
    (ScalarCartesianPropagator, ScalarSphericalPropagator),
    (VectorialCartesianPropagator, VectorialSphericalPropagator),
]


@pytest.mark.parametrize('n_defocus', [1, 3, 4])
def test_z_slices_are_spaced_by_defocus_step_and_centred(make_propagator, n_defocus):
    prop = make_propagator(ScalarCartesianPropagator, defocus_step=25, n_defocus=n_defocus)
    expected = (torch.arange(n_defocus) - n_defocus // 2) * 25.0
    assert torch.equal(prop.z, expected)
    assert prop.z[n_defocus // 2] == 0
    assert (prop.defocus_min, prop.defocus_max) == (float(expected[0]), float(expected[-1]))


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_stacks_with_nested_steps_share_their_slices(make_propagator, propagator_type):
    """z = -200, 0, 200 must give the same fields whether sampled with 3 or 5 slices."""
    coarse = make_propagator(propagator_type, defocus_step=200, n_defocus=3).compute_focus_field()
    fine = make_propagator(propagator_type, defocus_step=100, n_defocus=5).compute_focus_field()
    assert (coarse - fine[::2]).abs().max() < 1e-5 * fine.abs().max()


@pytest.mark.parametrize('n_pix_psf', [8, 9])
def test_psf_grid_is_pixel_centred(make_propagator, n_pix_psf):
    prop = make_propagator(ScalarCartesianPropagator, n_pix_psf=n_pix_psf, pix_size=50)
    assert torch.equal(prop.x, (torch.arange(n_pix_psf) - n_pix_psf // 2) * 50.0)
    assert prop.x[n_pix_psf // 2] == 0


def _airy_intensity(x, wavelength, na):
    """Normalized Airy pattern ``(2 J1(v) / v)^2`` on the square grid spanned by the 1-D coordinates ``x``."""
    xx, yy = torch.meshgrid(x, x, indexing='ij')
    v = 2 * math.pi / wavelength * na * torch.sqrt(xx ** 2 + yy ** 2)
    amplitude = torch.where(v > 1e-6, 2 * bessel_j1(v) / torch.where(v > 1e-6, v, 1.0), 1 - v ** 2 / 8)
    return amplitude ** 2


@pytest.mark.parametrize('propagator_type, low_na_kwargs', [
    (ScalarCartesianPropagator, {'sz_correction': False}),
    (ScalarSphericalPropagator, {'cos_factor': True}),
])
def test_low_na_psf_matches_airy_disk_sampled_every_pix_size(propagator_type, low_na_kwargs):
    """The analytic Airy disk only matches when its grid has the same pitch as the propagator's."""
    n_pix_psf, pix_size, wavelength, na = 65, 50.0, 500.0, 0.5
    prop = propagator_type(n_pix_pupil=401, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na,
                           pix_size=pix_size, n_i=1.0, n_i0=1.0, n_s=1.0, **low_na_kwargs)
    intensity = prop.compute_focus_field()[0, 0].abs() ** 2
    intensity = intensity / intensity.max()

    on_pixel_grid = _airy_intensity(prop.x, wavelength, na)
    stretched = torch.linspace(-prop.fov / 2, prop.fov / 2, n_pix_psf)  # pitch pix_size * n / (n - 1)
    on_stretched_grid = _airy_intensity(stretched, wavelength, na)

    error = (intensity - on_pixel_grid).abs().max()
    error_stretched = (intensity - on_stretched_grid).abs().max()
    assert error < 2e-3
    assert error < error_stretched / 5


def test_tilt_shifts_the_psf_by_a_known_number_of_pixels():
    """A Zernike tilt of ``c`` radians translates the Cartesian PSF by ``-c / (k n s_max)``.

    Checks the OSA ordering (index 1 is the vertical tilt, index 2 the horizontal one) and the
    pixel pitch at once: the shift is only an integer number of pixels if the pitch is ``pix_size``.
    """
    n_pix, pix_size, wavelength, na, n_i = 65, 30.0, 632.0, 1.2, 1.5
    shift_pixels = 3
    coefficient = shift_pixels * (2 * math.pi / wavelength) * n_i * (na / n_i) * pix_size
    centre = n_pix // 2
    for index, expected_peak in [(1, (centre - shift_pixels, centre)), (2, (centre, centre - shift_pixels))]:
        coefficients = [0.0] * 3
        coefficients[index] = coefficient
        prop = ScalarCartesianPropagator(n_pix_pupil=n_pix, n_pix_psf=n_pix, wavelength=wavelength, na=na,
                                         pix_size=pix_size, n_i=n_i, n_i0=n_i, zernike_coefficients=coefficients)
        intensity = prop.compute_focus_field()[0, 0].abs() ** 2
        assert divmod(int(torch.argmax(intensity)), n_pix) == expected_peak


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
def test_even_grid_peaks_at_centre_pixel(make_propagator, propagator_type):
    n_pix_psf = 32
    in_focus = _in_focus_intensity(
        make_propagator(propagator_type, n_pix_psf=n_pix_psf, n_pix_pupil=31, n_defocus=1).compute_focus_field())
    centre = n_pix_psf // 2
    assert divmod(int(torch.argmax(in_focus)), n_pix_psf) == (centre, centre)
    assert in_focus[centre, centre] > 1.05 * in_focus[centre - 1, centre - 1]


@pytest.mark.parametrize('cartesian_type, spherical_type', PARAMETERISATION_PAIRS)
@pytest.mark.parametrize('n_pix_psf', [32, 33])
def test_cartesian_and_spherical_agree_in_phase(make_propagator, cartesian_type, spherical_type, n_pix_psf):
    """Beyond the intensity, the complex fields agree up to a global factor, on even and odd grids."""
    fields = []
    for propagator_type in (cartesian_type, spherical_type):
        field = make_propagator(propagator_type, n_pix_psf=n_pix_psf, n_defocus=1).compute_focus_field()[0]
        fields.append(field / field[0, n_pix_psf // 2, n_pix_psf // 2])
    cartesian, spherical = fields
    significant = spherical.abs() > 0.05 * spherical.abs().max()
    assert (cartesian / spherical)[significant].angle().abs().max() < 1e-3


# Parameters at which the difference between the two Zernike meshes is clearly visible: at na=1.4, n=1.5 the
# mid-pupil sample of the spherical mesh sits at rho = 0.607, not 0.5.
ABERRATED_KWARGS = dict(n_pix_pupil=201, n_pix_psf=65, pix_size=40, n_defocus=1)


@pytest.mark.parametrize('spherical_type', [ScalarSphericalPropagator, VectorialSphericalPropagator])
def test_spherical_zernike_modes_are_evaluated_at_the_radius_of_the_samples(make_propagator, spherical_type):
    """The pupil is sampled uniformly in theta, so sample i sits at rho = sin(theta_i) / sin(theta_max)."""
    prop = make_propagator(spherical_type, zernike_coefficients=[0, 0, 0, 0, 1.0], **ABERRATED_KWARGS)
    rho = prop.rho
    assert rho.shape == (prop.n_pix_pupil,)
    assert float(rho[0]) == 0.0
    assert float(rho[-1]) == 1.0  # exactly, thanks to the clamp
    assert torch.all(torch.diff(rho) > 0)
    expected = torch.sin(prop.thetas.cpu().double()) / float(prop.s_max)
    torch.testing.assert_close(rho, expected.clamp(0.0, 1.0), atol=1e-6, rtol=0)
    assert (rho - torch.linspace(0, 1, prop.n_pix_pupil, dtype=rho.dtype)).abs().max() > 0.1

    # Defocus (OSA index 4) at the edge of the pupil is Z_2^0(1) = 1: the outermost sample is not zeroed.
    torch.testing.assert_close(prop._zernike_basis[4][-1].cpu(), torch.tensor(1.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize('cartesian_type, spherical_type', PARAMETERISATION_PAIRS)
@pytest.mark.parametrize('osa_index', [4, 12])  # defocus and primary spherical
def test_cartesian_and_spherical_agree_with_aberrations(make_propagator, cartesian_type, spherical_type, osa_index):
    """An aberration described by Zernike coefficients must give the same PSF in both parameterisations."""
    coefficients = [0.0] * (osa_index + 1)
    coefficients[osa_index] = 1.5
    cartesian = make_propagator(cartesian_type, zernike_coefficients=coefficients, **ABERRATED_KWARGS)
    spherical = make_propagator(spherical_type, zernike_coefficients=coefficients, **ABERRATED_KWARGS)
    reference = _normalized_in_focus_intensity(cartesian.compute_focus_field())
    torch.testing.assert_close(_normalized_in_focus_intensity(spherical.compute_focus_field()), reference,
                               atol=5e-3, rtol=0)

    # Regression guard: evaluating the spherical modes on the equispaced radius (the behaviour before the fix)
    # samples the wavefront at the wrong place and visibly disagrees.
    uniform_basis = zernike_basis(len(coefficients), spherical.n_pix_pupil, 'spherical')
    spherical._zernike_aberrations = create_zernike_aberrations(
        torch.tensor(coefficients), spherical.n_pix_pupil, 'spherical', basis=uniform_basis).to(spherical.device)
    on_uniform_mesh = _normalized_in_focus_intensity(spherical.compute_focus_field())
    assert (on_uniform_mesh - reference).abs().max() > 2e-2
