"""
A collection of miscellaneous functions.

"""
import typing as tp

import numpy as np
import torch


def convert_tensor_to_array(input_data: tp.Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert input data to a numpy array properly.

    Parameters
    ----------
    input_data : torch.Tensor or np.ndarray
        input image

    Returns
    -------
    output : np.ndarray
        corresponding numpy array

    """
    if isinstance(input_data, torch.Tensor):
        return input_data.detach().clone().cpu().numpy()
    elif isinstance(input_data, np.ndarray):
        return input_data.copy()
    else:
        raise TypeError(f'Unrecognized type of input, should be a torch.Tensor or np.ndarray, not {type(input_data)}')
