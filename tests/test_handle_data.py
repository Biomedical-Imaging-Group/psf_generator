"""Tests for the saving and loading helpers of ``psf_generator.utils.handle_data``."""
import numpy as np
import pytest
import torch

from psf_generator.utils.handle_data import (
    load_from_npy,
    load_image,
    load_stats_from_csv,
    save_as_npy,
    save_image,
    save_stats_as_csv,
)


def _sample(shape, dtype):
    generator = np.random.default_rng(0)
    array = generator.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        array = array + 1j * generator.standard_normal(shape)
    return array.astype(dtype)


@pytest.mark.parametrize('shape, dtype', [
    ((3, 1, 8, 8), np.complex64),   # scalar PSF stack
    ((3, 3, 8, 8), np.complex64),   # vectorial PSF stack
    ((1, 1, 8, 8), np.complex64),   # single in-focus slice
    ((8, 8), np.float32),           # plain image
])
def test_image_round_trip_preserves_shape_and_values(tmp_path, shape, dtype):
    image = _sample(shape, dtype)
    filepath = str(tmp_path / 'image.tif')
    save_image(filepath, image)
    loaded = load_image(filepath)
    assert loaded.shape == image.shape
    assert loaded.dtype == image.dtype
    assert np.array_equal(loaded, image)


def test_image_round_trip_accepts_a_tensor(tmp_path):
    image = torch.randn(2, 3, 4, 4, dtype=torch.complex64)
    filepath = str(tmp_path / 'psf.tif')
    save_image(filepath, image)
    assert np.array_equal(load_image(filepath), image.numpy())


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / 'absent.tif'))
    with pytest.raises(FileNotFoundError):
        load_stats_from_csv(str(tmp_path / 'absent.csv'))


def test_savers_accept_a_bare_filename(tmp_path, monkeypatch):
    """A path without a directory must not fail on ``os.makedirs('')``."""
    monkeypatch.chdir(tmp_path)
    image = _sample((2, 1, 4, 4), np.complex64)
    save_image('psf.tif', image)
    assert np.array_equal(load_image('psf.tif'), image)

    save_as_npy('psf.npy', image)
    assert np.array_equal(load_from_npy('psf.npy'), image)

    save_stats_as_csv('stats.csv', [(1, 0.5), (2, 1.5)])
    assert load_stats_from_csv('stats.csv') == [(1, 0.5), (2, 1.5)]


def test_savers_create_missing_directories(tmp_path):
    image = _sample((4, 4), np.float32)
    filepath = str(tmp_path / 'a' / 'b' / 'image.tif')
    save_image(filepath, image)
    assert np.array_equal(load_image(filepath), image)
    save_as_npy(str(tmp_path / 'c' / 'image.npy'), image)
    save_stats_as_csv(str(tmp_path / 'd' / 'stats.csv'), [(0, 0.0)])
