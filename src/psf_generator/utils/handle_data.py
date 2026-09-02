"""
A collection of functions to handle loading and saving of data and image.

- `image` uses common image formats, e.g., `.tif`
- `npy` uses numpy data format `.npy` for images
- `csv` uses `.csv` for statistical data

Notes
-----
    `save_image` writes the array as it is, without reordering its axes: a PSF of shape
    `(n_defocus, n_channels, n_pix_psf, n_pix_psf)` is stored with that shape and `load_image` reads it back
    unchanged, dtype (`complex64` included) and values included.

"""
import csv
import os
import typing as tp

import numpy as np
import skimage.io as skio
import tifffile
import torch

from psf_generator.utils.misc import convert_tensor_to_array

#: Extensions written and read with `tifffile`, which preserves the layout of an arbitrary n-dimensional array.
_TIFF_EXTENSIONS = ('.tif', '.tiff')


def _ensure_parent_directory(filepath: str) -> None:
    """Create the directory of `filepath`, if the path has one (a bare filename has not)."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _is_tiff(filepath: str) -> bool:
    """Whether `filepath` names a TIFF file."""
    return filepath.lower().endswith(_TIFF_EXTENSIONS)


def load_image(filepath: str):
    """
    Load data from filepath.

    Parameters
    ----------
    filepath : str
        Path to the file.

    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'{filepath} does not exist')
    if _is_tiff(filepath):
        # skimage.io.imread applies a channel heuristic that moves an axis of length 3 or 4 to the end;
        # tifffile returns the array exactly as it was written by ``save_image``.
        return tifffile.imread(filepath)
    return skio.imread(filepath)


def save_image(filepath: str, image: tp.Union[torch.Tensor, np.ndarray]):
    """
    Save image in specified format to specified location.

    Parameters
    ----------
    filepath : str
        Path to save the file.
    image : torch.Tensor or np.ndarray
        Image to be saved.

    Notes
    -----
    The array is written as it is: its axes are not reordered and its shape is preserved, so a PSF of shape
    `(n_defocus, n_channels, n_pix_psf, n_pix_psf)` is stored with that shape and read back unchanged by
    :func:`load_image`. TIFF files are written with `tifffile` and an explicit ``photometric='minisblack'``
    layout; without it tifffile guesses the meaning of the axes and either refuses to write a stack with a
    single channel or stores a three-channel stack as planar RGB, which comes back transposed.
    """
    image = convert_tensor_to_array(image)
    _ensure_parent_directory(filepath)
    if _is_tiff(filepath):
        tifffile.imwrite(filepath, image, photometric='minisblack')
    else:
        skio.imsave(filepath, image, check_contrast=False)


def save_as_npy(filepath: str, input_data: tp.Union[torch.Tensor, np.ndarray]):
    """
    Save data as a numpy array in a .npy file.

    Parameters
    ----------
    filepath : str
        Path to save the file.
    input_data : torch.Tensor or np.ndarray
        Data to be saved

    """
    input_data = convert_tensor_to_array(input_data)
    _ensure_parent_directory(filepath)
    np.save(filepath, input_data)

def load_from_npy(filepath: str) -> np.ndarray:
    """
    Load numpy array from a file.

    Parameters
    ----------
    filepath : str
        Path to file.

    Returns
    -------
    output : np.ndarray
        Loaded array.
    """
    return np.load(filepath)


def save_stats_as_csv(filepath: str, data: list):
    """
    Save statistical data to a csv file for further analysis or plotting.

    Statistical data such as the runtime values is saved as a list of tuples (index, value).

    Parameters
    ----------
    filepath : str
        Path to the file to store the statistics.
    data : list
        Statistics to be saved.

    """
    _ensure_parent_directory(filepath)
    with open(filepath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            writer.writerow(row)


def load_stats_from_csv(filepath: str):
    """
    Load data from a csv file.

    Parameters
    ----------
    filepath: str
        Path to the csv file.

    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'File {filepath} does not exist')

    with open(filepath, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in reader:
            data.append((int(row[0]), float(row[1])))
    return data
