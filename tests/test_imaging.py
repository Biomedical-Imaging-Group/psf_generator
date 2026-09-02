"""Tests of the dipole imagers (the detection path of the microscope).

The physics is checked against independent references rather than stored arrays:

* in a homogeneous medium the image of a dipole equals the apodized focus field of the vectorial propagators
  (reciprocity), component by component;
* the spherical (Bessel) and Cartesian (chirp Z) imagers, two independent evaluations of the same integral, agree;
* the energy in the image equals the power radiated by the dipole into the collection cone (Parseval), with the
  Fresnel transmittance of the coverslip when the dipole sits in water;
* the image of a displaced dipole is centred on the dipole.
"""
import json
import math

import pytest
import torch

from psf_generator.imaging import IMAGERS, CartesianDipoleImager, DipoleImager, SphericalDipoleImager
from psf_generator.propagators import VectorialCartesianPropagator, VectorialSphericalPropagator

IMAGER_TYPES = [SphericalDipoleImager, CartesianDipoleImager]

SMALL_KWARGS = dict(n_pix_pupil=63, n_pix_psf=63, wavelength=632, na=1.4, pix_size=100)
HOMOGENEOUS = dict(n_s=1.5, n_g=1.5, n_g0=1.5, n_i=1.5, n_i0=1.5)
STRATIFIED = dict(n_s=1.33, n_g=1.5, n_g0=1.5, n_i=1.5, n_i0=1.5)


def _fit_constant(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex constant c minimising ||a - c b||."""
    return (b.conj() * a).sum() / (b.abs() ** 2).sum()


def _intensity(field: torch.Tensor) -> torch.Tensor:
    return (field.abs() ** 2).sum(dim=1)


@pytest.mark.parametrize('imager_type', IMAGER_TYPES)
def test_image_shape_and_axial_component(imager_type):
    imager = imager_type(**SMALL_KWARGS, **STRATIFIED)
    field = imager.compute_image()
    assert tuple(field.shape) == (1, 3, 63, 63)
    assert field.dtype == torch.complex64
    assert torch.isfinite(field).all()
    assert field[:, :2].abs().max() > 0
    # the tube lens has a low NA: no axial component in the image plane
    assert torch.equal(field[:, 2], torch.zeros_like(field[:, 2]))


@pytest.mark.parametrize('imager_type', IMAGER_TYPES)
def test_positions_are_batched_consistently(imager_type):
    imager = imager_type(**SMALL_KWARGS, **STRATIFIED)
    positions = [(0.0, 0.0, 0.0), (200.0, 100.0, 300.0), (0.0, 0.0, 500.0), (200.0, 100.0, -300.0)]
    batch = imager.compute_image(positions=positions)
    assert tuple(batch.shape) == (4, 3, 63, 63)
    for index, position in enumerate(positions):
        single = imager.compute_image(positions=position)
        assert torch.allclose(batch[index], single[0], atol=1e-6 * batch.abs().max())
    # a tensor of positions is accepted too
    assert torch.equal(imager.compute_image(positions=torch.tensor(positions)), batch)


@pytest.mark.parametrize('imager_type, propagator_type', [
    (SphericalDipoleImager, VectorialSphericalPropagator),
    (CartesianDipoleImager, VectorialCartesianPropagator),
])
def test_homogeneous_medium_image_equals_apodized_focus_field(imager_type, propagator_type):
    """Reciprocity: without interfaces the image of a dipole is the focus field with the apodization factor."""
    z = 400.0
    imager = imager_type(**SMALL_KWARGS, **HOMOGENEOUS)
    image = imager.compute_image(dipole=(1.0, 0.0, 0.0), positions=[(0.0, 0.0, -z), (0.0, 0.0, 0.0), (0.0, 0.0, z)])
    propagator = propagator_type(**SMALL_KWARGS, **HOMOGENEOUS, e0x=1.0, e0y=0.0, apod_factor=True,
                                 defocus_step=z, n_defocus=3)
    focus = propagator.compute_focus_field()
    scale = _fit_constant(image[:, :2], focus[:, :2])
    difference = (image[:, :2] - scale * focus[:, :2]).abs().max()
    assert difference < 1e-3 * image.abs().max()
    # the focus field has an axial component that the image has not
    assert focus[:, 2].abs().max() > 1e-2 * focus.abs().max()


def test_spherical_and_cartesian_imagers_agree():
    """Two independent evaluations of the same integral; the residual is the discretisation of the rim of the
    pupil on the Cartesian grid, which decreases as 1 / n_pix_pupil."""
    kwargs = dict(SMALL_KWARGS, **STRATIFIED, z_focus=500.0)
    dipole = (1.0, 0.3j, 0.5)
    positions = [(150.0, -80.0, 300.0), (0.0, 0.0, 900.0)]
    spherical = SphericalDipoleImager(**kwargs).compute_image(dipole, positions)
    coarse = CartesianDipoleImager(**kwargs).compute_image(dipole, positions)
    fine = CartesianDipoleImager(**{**kwargs, 'n_pix_pupil': 255}).compute_image(dipole, positions)
    scale = spherical.abs().max()
    assert (spherical - coarse).abs().max() < 5e-2 * scale
    assert (spherical - fine).abs().max() < 1e-2 * scale
    assert (spherical - fine).abs().max() < 0.5 * (spherical - coarse).abs().max()


@pytest.mark.parametrize('imager_type', IMAGER_TYPES)
def test_image_is_centred_on_the_dipole(imager_type):
    imager = imager_type(**SMALL_KWARGS, **STRATIFIED)
    x_p, y_p = 300.0, -500.0  # multiples of the pixel size
    intensity = _intensity(imager.compute_image(positions=(x_p, y_p, 0.0)))[0]
    n = intensity.shape[-1]
    peak = divmod(int(torch.argmax(intensity)), n)
    # tensors are indexed [y, x]
    assert peak == (n // 2 + int(y_p / imager.pix_size), n // 2 + int(x_p / imager.pix_size))


def test_collected_energy_matches_the_dipole_radiation_pattern():
    """Parseval: the energy in the image is the power radiated into the collection cone (homogeneous medium)."""
    pix_size = 50.0
    imager = SphericalDipoleImager(n_pix_pupil=201, n_pix_psf=401, wavelength=500, na=1.0, pix_size=pix_size,
                                   **HOMOGENEOUS)
    c = math.cos(math.asin(1.0 / 1.5))
    axial = 2 * math.pi * (2 / 3 - c + c ** 3 / 3)          # integral of sin^2 over the cone
    transverse = 2 * math.pi * (1 - c) - axial / 2           # integral of 1 - sin^2 cos^2(phi)
    for dipole, expected in [((1.0, 0.0, 0.0), transverse), ((0.0, 0.0, 1.0), axial)]:
        energy = float((imager.compute_image(dipole).abs() ** 2).sum()) * pix_size ** 2
        assert energy == pytest.approx(expected, rel=0.05)


def test_energy_through_the_coverslip_uses_the_reciprocal_fresnel_coefficients():
    """The far field of a dipole in water seen from the glass is t_{s->g} n_g cos(theta_g) / (n_s cos(theta_s))
    times its far field in water, i.e. the reverse coefficient t_{g->s}: the power collected in the immersion
    medium (n_i times the energy of the image, the Poynting vector being proportional to n |E|^2) must equal the
    power radiated into the cone in water (n_s times the field energy) times the Fresnel transmittance."""
    pix_size = 50.0
    kwargs = dict(n_pix_pupil=201, n_pix_psf=401, wavelength=500, na=1.2, pix_size=pix_size, **STRATIFIED)

    def energy(fresnel):
        image = SphericalDipoleImager(**kwargs, fresnel=fresnel).compute_image((1.0, 0.0, 0.0))
        return float((image.abs() ** 2).sum()) * pix_size ** 2

    n_s, n_g = STRATIFIED['n_s'], STRATIFIED['n_g']
    theta_s = torch.linspace(0.0, math.asin(1.2 / n_s), 4001, dtype=torch.float64)
    sin_s, cos_s = torch.sin(theta_s), torch.cos(theta_s)
    cos_g = torch.sqrt(1 - (n_s * sin_s / n_g) ** 2)
    t_s = 2 * n_s * cos_s / (n_s * cos_s + n_g * cos_g)
    t_p = 2 * n_s * cos_s / (n_g * cos_s + n_s * cos_g)
    transmittance_ratio = n_g * cos_g / (n_s * cos_s)
    # azimuthal integral of |E_p|^2 = cos^2(theta_s) cos^2(phi) and |E_s|^2 = sin^2(phi) for an x dipole
    integrand = math.pi * (cos_s ** 2 * t_p ** 2 + t_s ** 2) * transmittance_ratio * sin_s
    expected = float(torch.trapezoid(integrand, theta_s)) * n_s / STRATIFIED['n_i']

    assert energy('reciprocal') == pytest.approx(expected, rel=0.05)
    # the forward coefficients alone miss the geometric factor and underestimate the collected light
    assert energy('forward') < 0.85 * expected


def test_supercritical_components_decay_with_the_height_of_the_dipole():
    """With NA > n_s the pupil extends beyond the critical angle; those (evanescent) components are collected
    from a dipole at the coverslip and vanish as it moves away."""
    imager = CartesianDipoleImager(**SMALL_KWARGS, **STRATIFIED)
    s = torch.linspace(-1, 1, imager.n_pix_pupil, dtype=torch.float64)
    s_yy, s_xx = torch.meshgrid(s, s, indexing='ij')
    sin_t = imager.s_max * torch.sqrt(s_xx ** 2 + s_yy ** 2)
    n_s, n_i = STRATIFIED['n_s'], STRATIFIED['n_i']
    supercritical = (sin_t > n_s / n_i) & (sin_t <= imager.s_max)
    assert supercritical.any()
    z_p = 400.0
    at_interface = imager.get_pupil(positions=(0.0, 0.0, 0.0))[0, 0].abs()
    away = imager.get_pupil(positions=(0.0, 0.0, z_p))[0, 0].abs()
    assert (at_interface[supercritical] > 0).all()
    # evanescent decay exp(-k z_p sqrt(n_i^2 sin^2 - n_s^2)) of every supercritical component
    decay = torch.exp(-imager.k * z_p * torch.sqrt((n_i * sin_t) ** 2 - n_s ** 2)[supercritical])
    assert torch.allclose((away / at_interface)[supercritical], decay.to(torch.float32), rtol=1e-3)
    assert decay.min() < 0.5 < decay.max()
    assert torch.isfinite(imager.compute_image(positions=[(0.0, 0.0, 0.0), (0.0, 0.0, z_p)])).all()


@pytest.mark.parametrize('imager_type', IMAGER_TYPES)
def test_zernike_aberrations_of_the_detection_path(imager_type):
    imager = imager_type(**SMALL_KWARGS, **STRATIFIED)
    reference = imager.compute_image()
    imager.update_zernike_coefficients([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])  # primary spherical
    aberrated = imager.compute_image()
    assert not torch.allclose(reference, aberrated, atol=1e-3 * reference.abs().max())
    assert imager.to_dict()['zernike_coefficients'][12] == 1.0
    if imager_type is SphericalDipoleImager:
        with pytest.warns(UserWarning, match='not axisymmetric'):
            imager.update_zernike_coefficients([0, 0, 0, 0, 0, 0, 0, 0.5])


@pytest.mark.parametrize('imager_type', IMAGER_TYPES)
def test_parameters_round_trip(imager_type, tmp_path):
    original = imager_type(**SMALL_KWARGS, **STRATIFIED, z_focus=750.0, zernike_coefficients=[0, 0, 0, 0, 0.2],
                           fresnel='forward')
    parameters = original.to_dict()
    json.dumps(parameters)
    assert parameters['imager'] == imager_type.get_name()
    restored = DipoleImager.from_dict(parameters)
    assert type(restored) is imager_type
    assert restored.to_dict() == parameters
    positions = [(50.0, 0.0, 200.0)]
    assert torch.equal(restored.compute_image(positions=positions), original.compute_image(positions=positions))
    path = tmp_path / 'imager.json'
    original.save_parameters(str(path))
    assert imager_type.load_parameters(str(path)).to_dict() == parameters


def test_registry_and_name_validation():
    assert set(IMAGERS) == {'spherical', 'cartesian'}
    parameters = SphericalDipoleImager(**SMALL_KWARGS).to_dict()
    with pytest.raises(ValueError, match='not'):
        CartesianDipoleImager.from_dict(parameters)
    with pytest.raises(ValueError, match='Unknown imager'):
        DipoleImager.from_dict({**parameters, 'imager': 'bessel'})
    with pytest.raises(ValueError, match="'imager' key"):
        DipoleImager.from_dict({key: value for key, value in parameters.items() if key != 'imager'})


@pytest.mark.parametrize('imager_type', IMAGER_TYPES)
@pytest.mark.parametrize('invalid, message', [
    ({'fresnel': 'exact'}, 'fresnel'),
    ({'na': 1.6, 'n_i0': 1.5}, 'numerical aperture'),
    ({'device': 'gpu'}, 'Invalid device'),
    ({'z_focus': 'top'}, 'z_focus'),
    ({'n_pix_pupil': 1}, 'n_pix_pupil'),
    ({'pix_size': 0}, 'pix_size'),
    ({'n_s': 0}, 'n_s'),
])
def test_invalid_parameters_are_rejected(imager_type, invalid, message):
    with pytest.raises(ValueError, match=message):
        imager_type(**{**SMALL_KWARGS, **invalid})


def test_invalid_dipoles_and_positions_are_rejected():
    imager = SphericalDipoleImager(**SMALL_KWARGS)
    with pytest.raises(ValueError, match='three components'):
        imager.compute_image(dipole=(1.0, 0.0))
    with pytest.raises(ValueError, match='positions'):
        imager.compute_image(positions=(0.0, 0.0))
    with pytest.raises(ValueError, match='positions'):
        imager.compute_image(positions=[[0.0, 0.0, 0.0, 0.0]])


def _accelerator() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return ''


@pytest.mark.skipif(not _accelerator(), reason='no non-CPU device available')
@pytest.mark.parametrize('imager_type', IMAGER_TYPES)
def test_imager_runs_on_an_accelerator(imager_type):
    device = _accelerator()
    kwargs = dict(n_pix_pupil=33, n_pix_psf=32, wavelength=632, na=1.4, pix_size=100, z_focus=300.0,
                  zernike_coefficients=[0, 0, 0, 0, 0.3], **STRATIFIED)
    positions = [(0.0, 0.0, 0.0), (120.0, -40.0, 500.0)]
    on_device = imager_type(device=device, **kwargs).compute_image(positions=positions)
    on_cpu = imager_type(device='cpu', **kwargs).compute_image(positions=positions)
    assert on_device.device.type == device
    assert torch.isfinite(on_device).all()
    assert (on_device.cpu() - on_cpu).abs().max() < 1e-4 * on_cpu.abs().max()
