import csv
import os
import typing as tp

import numpy as np
import skimage.io as skio
import torch

from utils.misc import convert_tensor_to_array


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


def save_image(image: tp.Union[torch.Tensor, np.ndarray], filepath: str):
    """
    Save image in specified format to specified location.

    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        Image to be saved.
    filepath : str
        Path to save the file.

    """
    image = convert_tensor_to_array(image)
    filepath = os.path.join(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    skio.imsave(filepath, image, check_contrast=False)


def save_stats_as_csv(data: list, filepath: str):
    """
    Save statistical data to a csv file for further analysis or plotting.

    Statistical data such as the runtime values is saved as a list of tuples (index, value).

    Parameters
    ----------
    data : list
        Statistics to be saved.
    filepath : str
        Path to the file to store the statistics.

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

    with open(filepath, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in reader:
            data.append((row[0], row[1]))
    return data