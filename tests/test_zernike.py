"""Tests for the OSA-indexed Zernike basis and the pupil phase aberrations built from it."""
import warnings

import pytest
import torch

from psf_generator.utils.zernike import (
    create_pupil_mesh,
    create_zernike_aberrations,
    nl_to_osa_index,
    osa_index_to_nl,
    zernike_basis,
    zernike_polynomial,
)

# (j, n, l) of the first modes, from the OSA/ANSI standard ordering.
OSA_TABLE = [
    (0, 0, 0), (1, 1, -1), (2, 1, 1),
    (3, 2, -2), (4, 2, 0), (5, 2, 2),
    (6, 3, -3), (7, 3, -1), (8, 3, 1), (9, 3, 3),
    (10, 4, -4), (11, 4, -2), (12, 4, 0), (13, 4, 2), (14, 4, 4),
]

N_PIX = 65


@pytest.fixture(scope='module')
def polar():
    kx, ky = create_pupil_mesh(N_PIX)
    return kx, ky, torch.sqrt(kx ** 2 + ky ** 2), torch.atan2(ky, kx)


@pytest.mark.parametrize('index, n, l', OSA_TABLE)
def test_osa_index_matches_standard_table(index, n, l):
    assert osa_index_to_nl(index) == (n, l)
    assert nl_to_osa_index(n, l) == index


def test_osa_index_round_trips():
    for index in range(300):
        n, l = osa_index_to_nl(index)
        assert n >= 0 and abs(l) <= n and (n - l) % 2 == 0
        assert nl_to_osa_index(n, l) == index


def test_invalid_indices_raise():
    with pytest.raises(ValueError):
        osa_index_to_nl(-1)
    with pytest.raises(ValueError):
        nl_to_osa_index(2, 1)  # n - l odd
    with pytest.raises(ValueError):
        nl_to_osa_index(1, 2)  # |l| > n
    with pytest.raises(ValueError):
        zernike_polynomial(2, 1, torch.zeros(3), torch.zeros(3))


# Closed forms of a few modes as functions of (kx, ky, rho, phi).
CLOSED_FORMS = {
    0: lambda kx, ky, rho, phi: torch.ones_like(rho),          # piston
    1: lambda kx, ky, rho, phi: ky,                              # vertical tilt: rho sin(phi)
    2: lambda kx, ky, rho, phi: kx,                              # horizontal tilt: rho cos(phi)
    3: lambda kx, ky, rho, phi: 2 * kx * ky,                     # oblique astigmatism: rho^2 sin(2 phi)
    4: lambda kx, ky, rho, phi: 2 * rho ** 2 - 1,                # defocus
    5: lambda kx, ky, rho, phi: kx ** 2 - ky ** 2,               # vertical astigmatism: rho^2 cos(2 phi)
    8: lambda kx, ky, rho, phi: (3 * rho ** 2 - 2) * kx,         # horizontal coma: (3 rho^3 - 2 rho) cos(phi)
    12: lambda kx, ky, rho, phi: 6 * rho ** 4 - 6 * rho ** 2 + 1,  # primary spherical
}


@pytest.mark.parametrize('index', sorted(CLOSED_FORMS))
def test_modes_match_closed_forms(polar, index):
    kx, ky, rho, phi = polar
    n, l = osa_index_to_nl(index)
    expected = torch.where(rho <= 1, CLOSED_FORMS[index](kx, ky, rho, phi), torch.zeros_like(rho))
    torch.testing.assert_close(zernike_polynomial(n, l, rho, phi), expected, atol=1e-5, rtol=1e-5)


def test_modes_vanish_outside_the_unit_disk(polar):
    kx, ky, rho, phi = polar
    basis = zernike_basis(15, N_PIX)
    assert torch.all(basis[:, rho > 1] == 0)
    assert basis.abs().max() <= 1 + 1e-6


def test_cartesian_basis_is_stacked_in_osa_order(polar):
    kx, ky, rho, phi = polar
    basis = zernike_basis(7, N_PIX)
    assert basis.shape == (7, N_PIX, N_PIX)
    assert basis.dtype == torch.float32
    for index in range(7):
        n, l = osa_index_to_nl(index)
        torch.testing.assert_close(basis[index], zernike_polynomial(n, l, rho, phi), atol=1e-6, rtol=1e-6)


def test_single_mode_basis_is_still_a_stack():
    assert zernike_basis(1, 17).shape == (1, 17, 17)
    assert zernike_basis(1, 17, 'spherical').shape == (1, 17)
    with pytest.raises(ValueError):
        zernike_basis(0, 17)
    with pytest.raises(ValueError):
        zernike_basis(3, 17, 'polar')


def test_spherical_basis_holds_the_radial_profile_of_axisymmetric_modes():
    basis = zernike_basis(13, N_PIX, 'spherical')
    assert basis.shape == (13, N_PIX)
    rho = torch.linspace(0, 1, N_PIX)
    for index in (0, 4, 12):  # piston, defocus, primary spherical
        n, l = osa_index_to_nl(index)
        assert l == 0
        torch.testing.assert_close(basis[index], zernike_polynomial(n, 0, rho, torch.zeros_like(rho)),
                                   atol=1e-6, rtol=1e-6)
    for index in range(13):
        if osa_index_to_nl(index)[1] != 0:
            assert torch.all(basis[index] == 0)


def test_modes_are_orthogonal_on_the_disk():
    n_modes = 15
    basis = zernike_basis(n_modes, 201).to(torch.float64).reshape(n_modes, -1)
    gram = basis @ basis.T
    norms = torch.sqrt(torch.diag(gram))
    correlation = gram / (norms[:, None] * norms[None, :])
    off_diagonal = correlation - torch.eye(n_modes, dtype=correlation.dtype)
    assert off_diagonal.abs().max() < 0.02


def test_single_coefficient_gives_a_constant_phase_on_the_disk():
    aberration = create_zernike_aberrations([0.5], 17, 'cartesian')
    assert aberration.shape == (17, 17)
    assert aberration.dtype == torch.complex64
    kx, ky = create_pupil_mesh(17)
    inside = kx ** 2 + ky ** 2 <= 1
    torch.testing.assert_close(aberration.angle()[inside], torch.full((int(inside.sum()),), 0.5),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(aberration.abs(), torch.ones(17, 17), atol=1e-6, rtol=0)


def test_aberration_is_the_phase_of_the_weighted_sum():
    coefficients = torch.tensor([0.0, 0.3, -0.2, 0.0, 1.0])
    expected_phase = (coefficients[:, None, None] * zernike_basis(5, 33)).sum(dim=0)
    aberration = create_zernike_aberrations(coefficients, 33, 'cartesian')
    torch.testing.assert_close(aberration, torch.exp(1j * expected_phase).to(torch.complex64))


def test_precomputed_basis_gives_identical_result_and_is_validated():
    basis = zernike_basis(5, 33)
    with_basis = create_zernike_aberrations([0, 0, 0, 0, 1.0], 33, 'cartesian', basis=basis)
    without_basis = create_zernike_aberrations([0, 0, 0, 0, 1.0], 33, 'cartesian')
    assert torch.equal(with_basis, without_basis)
    with pytest.raises(ValueError):
        create_zernike_aberrations([1.0, 2.0], 33, 'cartesian', basis=basis)
    with pytest.raises(ValueError):
        create_zernike_aberrations([0, 0, 0, 0, 1.0], 17, 'cartesian', basis=basis)


def test_spherical_mesh_warns_about_and_ignores_non_axisymmetric_modes():
    with pytest.warns(UserWarning, match=r'modes \[5\] are not axisymmetric'):
        aberration = create_zernike_aberrations([0, 0, 0, 0, 0.7, 0.5], 17, 'spherical')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        expected = create_zernike_aberrations([0, 0, 0, 0, 0.7], 17, 'spherical')
    assert aberration.shape == (17,)
    torch.testing.assert_close(aberration, expected)


def test_gradient_flows_to_the_coefficients():
    coefficients = torch.full((6,), 0.1, requires_grad=True)
    aberration = create_zernike_aberrations(coefficients, 17, 'cartesian')
    aberration.imag.sum().backward()
    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()
    assert coefficients.grad[0] != 0


def test_spherical_basis_can_be_sampled_at_given_radii():
    """The spherical propagators sample the pupil uniformly in theta and pass their own radii."""
    thetas = torch.linspace(0, torch.arcsin(torch.tensor(1.4 / 1.5)), N_PIX, dtype=torch.float64)
    rho = torch.sin(thetas) / torch.sin(thetas[-1])
    basis = zernike_basis(13, N_PIX, 'spherical', rho=rho)
    assert basis.shape == (13, N_PIX)
    assert basis.dtype == torch.float32
    for index in range(13):
        n, l = osa_index_to_nl(index)
        if l == 0:
            expected = zernike_polynomial(n, 0, rho, torch.zeros_like(rho)).to(torch.float32)
            torch.testing.assert_close(basis[index], expected, atol=1e-6, rtol=1e-6)
        else:
            assert torch.all(basis[index] == 0)
    # The non-uniform mesh really differs from the default equispaced one.
    assert (basis - zernike_basis(13, N_PIX, 'spherical')).abs().max() > 0.1


def test_spherical_basis_validates_the_given_radii():
    with pytest.raises(ValueError, match='shape'):
        zernike_basis(5, N_PIX, 'spherical', rho=torch.linspace(0, 1, N_PIX + 1))
    with pytest.raises(ValueError, match='shape'):
        zernike_basis(5, N_PIX, 'spherical', rho=torch.zeros(1, N_PIX))
    with pytest.raises(ValueError, match='spherical'):
        zernike_basis(5, N_PIX, 'cartesian', rho=torch.linspace(0, 1, N_PIX))
