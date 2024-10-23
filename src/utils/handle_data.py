import os
import typing as tp

import numpy as np
import skimage.io as skio
import torch

from utils.misc import convert_tensor_to_array


def load_data(filepath: str):
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


def save_image(image: tp.Union[torch.Tensor, np.ndarray], filename: str, path: str, filetype: str = '.tif'):
    """
    Save image in specified format to specified location.

    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        Image to be saved.
    filename : str
        Name of the file.
    path: str
        Location of the file.
    filetype: str
        Format of the file. Default is '.tif'.

    """
    image = convert_tensor_to_array(image)
    filepath = os.path.join(path, filename + filetype)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    skio.imsave(filepath, image, check_contrast=False)

