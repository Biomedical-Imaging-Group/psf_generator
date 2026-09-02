"""Tests for the plotting helpers of ``psf_generator.utils.plots``."""
import os

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

from psf_generator.utils.plots import apply_disk_mask, plot_psf, plot_pupil  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_plot_pupil_saves_to_a_bare_filename(tmp_path, monkeypatch):
    """A path without a directory must not fail on ``os.makedirs('')``."""
    monkeypatch.chdir(tmp_path)
    plot_pupil(torch.ones(1, 1, 8, 8, dtype=torch.complex64), 'scalar_cartesian', filepath='pupil.png')
    assert os.path.isfile('pupil.png')


def test_plot_psf_saves_to_a_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plot_psf(torch.ones(1, 1, 8, 8, dtype=torch.complex64), 'scalar_cartesian', filepath='psf.png')
    assert os.path.isfile('psf.png')


def test_plots_create_missing_directories(tmp_path):
    filepath = str(tmp_path / 'a' / 'b' / 'psf.png')
    plot_psf(torch.ones(3, 1, 8, 8, dtype=torch.complex64), 'scalar_cartesian', filepath=filepath)
    assert os.path.isfile(filepath)


def test_disk_mask_is_symmetric_and_keeps_the_centre():
    image = np.ones((9, 9))
    masked = apply_disk_mask(image)
    assert masked[4, 4] == 1
    assert np.all(np.isnan(masked[[0, 0, -1, -1], [0, -1, 0, -1]]))  # the four corners
    kept = ~np.isnan(masked)
    assert np.array_equal(kept, kept[::-1, :])
    assert np.array_equal(kept, kept[:, ::-1])
