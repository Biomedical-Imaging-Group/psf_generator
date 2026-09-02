"""Tests of the modalities: iSCAT, COBRI and dark-field microscopy of a Rayleigh scatterer."""
import json
import math

import pytest
import torch

from psf_generator.imaging import CartesianDipoleImager, SphericalDipoleImager
from psf_generator.modalities import (
    MODALITIES,
    COBRIMicroscope,
    DarkFieldMicroscope,
    ISCATMicroscope,
    Modality,
    Particle,
    ScatteringMicroscope,
)

# A 30 nm gold nanoparticle imaged at 517.5 nm with an oil objective focused 1 um above the coverslip, as in
# Dong et al. 2021 (appendix C); small grids keep the tests fast.
GOLD = Particle(radius=15.0, permittivity=-3.7328 + 2.7725j)
SETUP = dict(wavelength=517.5, na=1.3, n_s=1.33, n_g=1.5, n_g0=1.5, n_i=1.5, n_i0=1.5,
             n_pix_pupil=63, n_pix_psf=63, pix_size=40, z_focus=1000.0)
POSITIONS = [(0.0, 0.0, 900.0), (80.0, -120.0, 1300.0)]


def test_particle_polarizability_cross_section_and_mass():
    n_s = 1.33
    alpha = GOLD.polarizability(n_s)
    expected = 4 * math.pi * 15.0 ** 3 * (GOLD.permittivity - n_s ** 2) / (GOLD.permittivity + 2 * n_s ** 2)
    assert alpha == pytest.approx(expected)
    k_s = 2 * math.pi * n_s / 517.5
    assert GOLD.scattering_cross_section(517.5, n_s) == pytest.approx(k_s ** 4 * abs(alpha) ** 2 / (6 * math.pi))

    dielectric = Particle(10.0, refractive_index=1.5)
    assert dielectric.permittivity == 2.25
    assert dielectric.polarizability(1.5) == 0
    # a 66 kDa protein at 1.35 g/cm^3 has a volume of 81 nm^3, i.e. a radius of 2.7 nm
    protein = Particle.from_mass(66.0, density=1.35, refractive_index=1.46)
    assert protein.radius == pytest.approx(2.69, abs=0.01)

    for kwargs in ({}, {'refractive_index': 1.5, 'permittivity': 2.25}):
        with pytest.raises(ValueError, match='exactly one'):
            Particle(10.0, **kwargs)
    with pytest.raises(ValueError, match='radius'):
        Particle(-1.0, refractive_index=1.5)
    for particle in (GOLD, dielectric):
        assert Particle.from_dict(json.loads(json.dumps(particle.to_dict()))) == particle


def test_reference_intensity_of_each_scheme():
    n_s, n_g = SETUP['n_s'], SETUP['n_g']
    reflectance = ((n_g - n_s) / (n_g + n_s)) ** 2
    transmittance = (2 * n_s / (n_s + n_g)) ** 2
    assert ISCATMicroscope(GOLD, **SETUP).reference_intensity == pytest.approx(reflectance, rel=1e-5)
    assert ISCATMicroscope(GOLD, attenuation=0.5, **SETUP).reference_intensity == pytest.approx(reflectance / 4, rel=1e-5)
    assert COBRIMicroscope(GOLD, **SETUP).reference_intensity == pytest.approx(transmittance, rel=1e-5)
    assert DarkFieldMicroscope(GOLD, **SETUP).reference_intensity == 0.0
    # the reference keeps the polarization of the illumination
    reference = ISCATMicroscope(GOLD, e0x=0.0, e0y=1.0, **SETUP).reference_field()
    assert reference[0] == 0 and reference[2] == 0 and abs(reference[1]) ** 2 == pytest.approx(reflectance, rel=1e-5)


def test_image_is_the_interference_of_reference_and_scattered_fields():
    microscope = ISCATMicroscope(GOLD, **SETUP)
    reference, scattered = microscope.compute_fields(POSITIONS)
    assert tuple(scattered.shape) == (2, 3, 63, 63)
    image = microscope.compute_image(POSITIONS)
    expected = ((reference[None, :, None, None] + scattered).abs() ** 2).sum(dim=1)
    assert torch.allclose(image, expected)
    contrast = microscope.compute_contrast(POSITIONS)
    background = microscope.reference_intensity
    assert torch.allclose(contrast, (image - background) / background)
    assert contrast.abs().max() > 1e-4


def test_darkfield_image_is_the_scattered_intensity():
    microscope = DarkFieldMicroscope(GOLD, **SETUP)
    scattered = microscope.scattered_field(POSITIONS)
    assert torch.allclose(microscope.compute_image(POSITIONS), (scattered.abs() ** 2).sum(dim=1))
    with pytest.raises(ValueError, match='reference'):
        microscope.compute_contrast(POSITIONS)
    with pytest.raises(ValueError, match='reference'):
        ISCATMicroscope(GOLD, attenuation=0.0, **SETUP).compute_contrast(POSITIONS)


def test_iscat_and_cobri_scattered_fields_differ_by_the_double_pass_phase():
    """The illumination reaches the particle from below (iSCAT) or from above (COBRI): the scattered fields
    differ by the transmission into the sample and the phase 2 k n_s z_p (plus the excess path Delta)."""
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 250.0), (60.0, 0.0, 700.0)]
    iscat = ISCATMicroscope(GOLD, **SETUP)
    cobri = COBRIMicroscope(GOLD, **SETUP)
    z_p = torch.tensor([position[2] for position in positions], dtype=torch.float64)
    n_s, n_g = SETUP['n_s'], SETUP['n_g']
    transmission = 2 * n_g / (n_g + n_s)
    assert iscat.excess_path() == pytest.approx(-SETUP['n_i'] ** 2 * SETUP['z_focus'] / n_s)
    phase = iscat.k * (iscat.excess_path() + 2 * n_s * z_p)
    ratio = (transmission * torch.exp(1j * phase)).to(torch.complex64)
    expected = cobri.scattered_field(positions) * ratio[:, None, None, None]
    actual = iscat.scattered_field(positions)
    assert (actual - expected).abs().max() < 1e-5 * actual.abs().max()


def test_contrast_is_linear_in_the_polarizability_of_small_particles():
    small = Particle(2.5, permittivity=GOLD.permittivity)
    large = Particle(5.0, permittivity=GOLD.permittivity)
    contrast_small = ISCATMicroscope(small, **SETUP).compute_contrast(POSITIONS)
    contrast_large = ISCATMicroscope(large, **SETUP).compute_contrast(POSITIONS)
    # the polarizability (and the contrast) scales with the volume; |E_sca|^2 adds a small quadratic term
    assert torch.allclose(contrast_large, 8 * contrast_small, atol=1e-2 * contrast_large.abs().max())


def test_spherical_and_cartesian_imagers_give_the_same_contrast():
    spherical = ISCATMicroscope(GOLD, imager='spherical', **SETUP).compute_contrast(POSITIONS)
    cartesian = ISCATMicroscope(GOLD, imager='cartesian', **SETUP).compute_contrast(POSITIONS)
    assert (spherical - cartesian).abs().max() < 1e-2 * spherical.abs().max()


def test_contrast_scales_from_gold_nanoparticles_to_single_proteins():
    """A 30 nm gold particle interfering with the glass-water reflection gives a contrast of tens of percent in
    focus; the polarizability of a 100 kDa protein is about 4000 times smaller, so its contrast is about 1e-4, in
    line with mass photometry."""
    microscope = ISCATMicroscope(GOLD, **SETUP)
    positions = [(0.0, 0.0, z) for z in range(0, 2001, 100)]
    contrast = microscope.compute_contrast(positions)
    on_axis = contrast[:, 63 // 2, 63 // 2]
    assert 0.1 < on_axis.abs().max() < 1.0
    protein = Particle.from_mass(100.0, density=1.35, refractive_index=1.46)
    protein_contrast = ISCATMicroscope(protein, **SETUP).compute_contrast(positions)[:, 63 // 2, 63 // 2]
    assert 3e-5 < protein_contrast.abs().max() < 1e-3
    # the sign of the on-axis signal oscillates with the height of the particle (period lambda / 2 n_s = 195 nm)
    assert (on_axis > 0).any() and (on_axis < 0).any()


def test_polarization_of_the_illumination_rotates_the_image():
    along_x = ISCATMicroscope(GOLD, e0x=1.0, e0y=0.0, **SETUP).compute_contrast((0.0, 0.0, 1200.0))[0]
    along_y = ISCATMicroscope(GOLD, e0x=0.0, e0y=1.0, **SETUP).compute_contrast((0.0, 0.0, 1200.0))[0]
    assert torch.allclose(along_y, along_x.T, atol=1e-5 * along_x.abs().max())


def test_imager_arguments():
    imager = CartesianDipoleImager(**SETUP)
    microscope = ISCATMicroscope(GOLD, imager=imager)
    assert microscope.imager is imager
    with pytest.raises(ValueError, match='already built'):
        ISCATMicroscope(GOLD, imager=imager, na=1.2)
    with pytest.raises(ValueError, match='Unknown imager'):
        ISCATMicroscope(GOLD, imager='bessel')
    with pytest.raises(ValueError, match='scheme'):
        ScatteringMicroscope(GOLD, scheme='brightfield')
    with pytest.raises(TypeError, match='Particle'):
        ISCATMicroscope('gold')
    with pytest.raises(ValueError, match='attenuation'):
        ISCATMicroscope(GOLD, attenuation=-0.1)


@pytest.mark.parametrize('modality_type', [ISCATMicroscope, COBRIMicroscope, DarkFieldMicroscope])
def test_parameters_round_trip(modality_type, tmp_path):
    original = modality_type(GOLD, e0x=0.6, e0y=0.8j, imager='cartesian', **SETUP)
    parameters = original.to_dict()
    json.dumps(parameters)
    assert parameters['modality'] == modality_type.get_name()
    assert parameters['imager']['imager'] == 'cartesian'
    for restored in (Modality.from_dict(parameters), modality_type.from_dict(parameters),
                     ScatteringMicroscope.from_dict(parameters)):
        assert type(restored) is modality_type
        assert restored.to_dict() == parameters
        assert isinstance(restored.imager, CartesianDipoleImager)
        assert torch.equal(restored.compute_image(POSITIONS), original.compute_image(POSITIONS))
    path = tmp_path / 'microscope.json'
    original.save_parameters(str(path))
    assert Modality.load_parameters(str(path)).to_dict() == parameters
    with pytest.raises(ValueError, match='not'):
        ISCATMicroscope.from_dict(COBRIMicroscope(GOLD, **SETUP).to_dict())


def test_generic_scattering_microscope_stores_its_scheme():
    generic = ScatteringMicroscope(GOLD, scheme='cobri', **SETUP)
    parameters = generic.to_dict()
    assert parameters['modality'] == 'scattering' and parameters['scheme'] == 'cobri'
    restored = Modality.from_dict(parameters)
    assert type(restored) is ScatteringMicroscope and restored.scheme == 'cobri'
    assert torch.equal(restored.compute_image(POSITIONS), COBRIMicroscope(GOLD, **SETUP).compute_image(POSITIONS))
    assert set(MODALITIES) == {'scattering', 'iscat', 'cobri', 'darkfield'}


def _accelerator() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return ''


@pytest.mark.skipif(not _accelerator(), reason='no non-CPU device available')
@pytest.mark.parametrize('imager', ['spherical', 'cartesian'])
def test_modality_runs_on_an_accelerator(imager):
    device = _accelerator()
    kwargs = dict(SETUP, n_pix_pupil=33, n_pix_psf=32, imager=imager)
    on_device = ISCATMicroscope(GOLD, device=device, **kwargs).compute_contrast(POSITIONS)
    on_cpu = ISCATMicroscope(GOLD, device='cpu', **kwargs).compute_contrast(POSITIONS)
    assert on_device.device.type == device
    assert (on_device.cpu() - on_cpu).abs().max() < 1e-4 * on_cpu.abs().max()


def test_imager_types_are_interchangeable_in_the_modality():
    for imager_type in (SphericalDipoleImager, CartesianDipoleImager):
        microscope = ISCATMicroscope(GOLD, imager=imager_type(**SETUP))
        assert tuple(microscope.compute_image(POSITIONS).shape) == (2, 63, 63)
