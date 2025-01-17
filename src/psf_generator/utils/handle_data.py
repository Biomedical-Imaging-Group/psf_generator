"""
A collection of functions to handle loading and saving of data and image.

- `image` uses common image formats, e.g., `.tif`
- `npy` uses numpy data format `.npy` for images
- `csv` uses `.csv` for statistical data

Notes
-----
    `save_image` follows convention (spatial dimensions, channels), i.e. it changes the axes of the input image.
    For tests, we save images in `.npy` format to avoid this inconvenience.

"""
import csv
import os
import typing as tp

import numpy as np
import skimage.io as skio
import torch

from psf_generator.utils.misc import convert_tensor_to_array


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
    Scikit-image and tifffile both follow the convention of putting the channel dimension after x and y.
    The saved tif image thus has dimension (z, x, y, channels) instead of (z, channels, x, y).
    """
    image = convert_tensor_to_array(image)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
