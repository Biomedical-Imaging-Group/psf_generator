import os
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.misc import convert_tensor_to_array

_FIG_SIZE = 6
_SUP_TITLE_SIZE = 18
_TITLE_SIZE = 16
_LABEL_SIZE = 14
_TICK_SIZE = 12
lw = 1
markersize = 6


def colorbar(mappable, cbar_ticks: tp.Union[str, tp.List, None] = 'auto'):
    """
    colorbar with the option to add or remove ticks
    Parameters
    ----------
    mappable
    cbar_ticks: None, or 'auto' or List of ticks.
        If None, no ticks visible;
        If 'auto': ticks are determined automatically
        otherwise, set the ticks as given by cbar_ticks
    """
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    if cbar_ticks == 'auto':
        pass
    elif cbar_ticks is None:
        cbar.set_ticks([])
    else:
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar_ticks], fontsize=_TICK_SIZE)
    plt.sca(last_axes)
    return cbar


def _compute_intensity_of_psf(input_image: np.ndarray) -> np.ndarray:
    """
    Compute the intensity of the PSF from the electric field.

    The input array must be 4D with this convention:
    - dim one: z axis, or defocus slices.
    - dim two: electric field components. Only one for scalar and three :math:`(e_x, e_y, e_z)` for vectorial.
    - dim three and four : (x, y) axes

    The intensity is computed as follows:
    .. math:: I = sqrt(sum_{i}^{n_e} |e_i(x, y, z)|^2).

    Parameters
    ----------
    input_image : np.ndarray
        Scalar or vectorial electric field. 4D array.
    Returns
    -------
    output : np.ndarray
        Intensity of the PSF. 3D array.

    """
    if input_image.ndim != 4:
        raise ValueError(f'The input image must be 4D instead of {input_image.ndim}')
    else:
        return np.sqrt(np.sum(np.abs(input_image[:, :, :, :]) ** 2, axis=1))


def plot_psf_intensity_maps(
        psf_image: tp.Union[torch.Tensor, np.ndarray],
        name_of_propagator: str,
        z_slice_number: int = None,
        x_slice_number: int = None,
        y_slice_number: int = None,
        cmap: str = 'viridis',
        filepath: str = None
):
    """
    Plot three orthogonal slices of the 3D intensity map of the PSF.

    Parameters
    ----------
    psf_image : torch.Tensor
        Computed PSF from one of the four propagators. Direct output of the propagator.
    name_of_propagator : str
        Name of the propagator.
    z_slice_number : int, optional
        Slice number at the z-axis for x-y planes. Default is the middle slice.
    x_slice_number : int, optional
        Slice number at the x-axis for y-z planes. Default is the middle slice.
    y_slice_number: int, optional
        Slice number at the y-axis for x-z planes. Default is the middle slice.
    cmap : str, optional
        colormap. Default is 'viridis'.
    filepath : str, optional
        Path to save the plot. Default is None and figure is not saved.

    """
    psf_array = convert_tensor_to_array(psf_image)
    psf_intensity = _compute_intensity_of_psf(psf_array)
    number_of_z_slices, number_of_electric_field_components, number_of_pixel_x, number_of_pixel_y = psf_image.shape
    if z_slice_number is None:
        z_slice_number = int(number_of_z_slices // 2)
    if x_slice_number is None:
        x_slice_number = int(number_of_pixel_x // 2)
    if y_slice_number is None:
        y_slice_number = int(number_of_pixel_y // 2)

    image_list = [psf_intensity[z_slice_number, :, :],
                  psf_intensity[:, x_slice_number, :].T,
                  psf_intensity[:, :, y_slice_number].T]
    nrows = 1
    ncols = 3
    titles = [f'x-y plane at z={z_slice_number}/{number_of_z_slices} slices',
              f'y-z plane at x={x_slice_number}/{number_of_pixel_x} slices',
              f'x-z plane at y={y_slice_number}/{number_of_pixel_y} slices']

    cbar_min = min([np.min(img) for img in image_list])
    cbar_max = max([np.max(img) for img in image_list])

    figure, axes = plt.subplots(nrows, ncols, figsize=(ncols * _FIG_SIZE, nrows * _FIG_SIZE))
    for ax, img, title in zip(axes, image_list, titles):
        norm = plt.Normalize(cbar_min, cbar_max)
        im = ax.imshow(img, norm=norm, cmap=cmap)
        colorbar(im, cbar_ticks=[cbar_min, cbar_max])
        ax.set_title(title, fontsize=_TITLE_SIZE)
        x_ticks = [0, img.shape[1]]
        xtick_labels = x_ticks
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(xtick_labels, fontsize=_TICK_SIZE)
        y_ticks = [0, img.shape[0]]
        ax.set_yticks(y_ticks)
        ytick_labels = y_ticks
        ax.set_yticklabels(ytick_labels, fontsize=_TICK_SIZE)
    plt.suptitle(f'PSF intensity maps at three orthogonal planes ({name_of_propagator})', fontsize=_SUP_TITLE_SIZE)
    figure.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()


def plot_benchmark_results(results: list, labels: list, title: str, filepath: str = None):
    """
    Plot results of the benchmarking.

    Parameters
    ----------
    results : list
        List that contains tuples of the variable to be benchmarked and the resulting value.
    labels : str
        Label/name of the data.
    title : str
        Title of the figure.
    filepath : str, optional
        Path to save the figure. Default is None and figure is not saved.

    """
    figure, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE*1.5, _FIG_SIZE))
    colors = ['red', 'red', 'blue', 'blue', 'orange', 'orange', 'green', 'green']
    for result, label, color in zip(results, labels, colors):
        x, y = zip(*result)
        if 'cpu' in label:
            ls = 'solid'
        else:
            ls = 'dashed'
        ax.loglog(x, y, base=2, label=label, ls=ls, marker='.', markersize=markersize, lw=lw, color=color)
    ax.legend(fontsize=_TICK_SIZE)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.set_xlabel('Numerical size of the pupil / pixels', fontsize=_LABEL_SIZE)
    ax.set_ylabel('Time / s', fontsize=_LABEL_SIZE)
    plt.grid(color='gray', ls='dotted', lw=lw)
    figure.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()

