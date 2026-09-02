"""Tests that every propagator runs on a non-CPU device and agrees with the CPU result.

The tests are skipped when no accelerator is available; on this machine they run on MPS, where every
computation is done in single precision, hence the loose (but still meaningful) tolerance.
"""
import pytest
import torch

from conftest import ALL_PROPAGATORS

# Option sets that each add a factor to the pupil; every one of them used to crash the spherical
# propagators on a non-CPU device because the correction factor was built from CPU tensors.
OPTION_SETS = [
    {},
    {'apod_factor': True},
    {'envelope': 0.8},
    {'gibson_lanni': True},
    {'cos_factor': True},              # spherical only
    {'zernike_coefficients': [0, 0, 0, 0, 0.5]},
    {'special_phase_mask': 'vortex'},  # Cartesian only
]

DEVICE_KWARGS = dict(n_pix_pupil=33, n_pix_psf=32, n_defocus=3, defocus_step=100)


def _accelerator() -> str:
    """Name of the first available non-CPU device."""
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return ''


DEVICE = _accelerator()

pytestmark = pytest.mark.skipif(not DEVICE, reason='no non-CPU device available')


@pytest.mark.parametrize('propagator_type', ALL_PROPAGATORS)
@pytest.mark.parametrize('options', OPTION_SETS, ids=lambda options: '-'.join(options) or 'plain')
def test_propagator_runs_on_an_accelerator(make_propagator, propagator_type, options):
    is_spherical = 'spherical' in propagator_type.get_name()
    if 'cos_factor' in options and not is_spherical:
        pytest.skip('cos_factor only exists on the spherical propagators')
    if 'special_phase_mask' in options and is_spherical:
        pytest.skip('special_phase_mask only exists on the Cartesian propagators')

    on_device = make_propagator(propagator_type, device=DEVICE, **DEVICE_KWARGS, **options).compute_focus_field()
    on_cpu = make_propagator(propagator_type, device='cpu', **DEVICE_KWARGS, **options).compute_focus_field()

    assert on_device.device.type == DEVICE
    assert torch.isfinite(on_device).all()
    difference = (on_device.cpu() - on_cpu).abs().max()
    assert difference < 1e-4 * on_cpu.abs().max()
