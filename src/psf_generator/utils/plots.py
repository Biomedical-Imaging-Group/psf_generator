"""
A collection of plotting functions.

"""
import os
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from misc import convert_tensor_to_array

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


def apply_disk_mask(img):
    """Apply a disk mask to a square image."""
    img = img.copy()
    # check if square
    if img.shape[0] != img.shape[1]:
        raise ValueError('Can not apply disk mask on a non-square image')
    # create mask
    size = img.shape[0]
    mask = np.zeros((size, size))
    i = np.linspace(0, size, size)
    j = np.linspace(0, size, size)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    disk = (ii - size / 2) ** 2 + (jj - size / 2) ** 2 <= (size / 2) ** 2
    mask[disk] = 1
    # apply mask, set values outside the mask to nan
    img = np.where(mask, img, np.nan)
    return img


def _compute_psf_intensity(input_image: np.ndarray) -> np.ndarray:
    """
    Compute the intensity of a complex field.

    The input array must be 4D with this convention:
    - dim one: z axis, or defocus slices.
    - dim two: electric field components. Only one for scalar and three :math:`(e_x, e_y, e_z)` for vectorial.
    - dim three and four : (x, y) axes

    The intensity is computed as follows:

    .. math:: I = \sum_{i=1}^{N_e} |e_i(x, y, z)|^2, \quad N_e = 1 \, \mathrm{or} \, 3.

    Parameters
    ----------
    input_image : np.ndarray
        Scalar or vectorial complex field. 4D array.

    Returns
    -------
    output : np.ndarray
        Intensity of the field. 4D array.

    """
    if input_image.ndim != 4:
        raise ValueError(f'The input image must be 4D instead of {input_image.ndim}')
    else:
        intensity = np.sum(np.abs(input_image) ** 2, axis=1)
        return intensity[:, np.newaxis, :, :]


def plot_pupil(
        pupil: tp.Union[torch.Tensor, np.ndarray],
        name_of_propagator: str,
        filepath: str = None
):
    """
    Plot the modulus and phase of a scalar or vectorial pupil for the Cartesian propagator.

    Parameters
    ----------
    pupil : torch.Tensor or np.ndarray
        Pupil image to plot.
    name_of_propagator : str
        Name of the propagator.
    filepath: str, optional
        Path to save the plot. Default is None, no file is saved.

    """
    # convert to numpy array
    pupil_array = convert_tensor_to_array(pupil)
    # compute modulus and phase
    pupil_modulus = np.abs(pupil_array)
    pupil_phase = np.angle(pupil_array)
    pupil_list = [pupil_modulus, pupil_phase]

    if pupil_array.ndim == 2:
        nrows = 1
        pupil_list = [x[np.newaxis, :, :] for x in pupil_list]
        row_titles = ['']
    elif pupil_array.ndim == 3:
        nrows = pupil_array.shape[0]
        row_titles = [r'$E_x$', r'$E_y$', r'$E_z$']
    else:
        raise ValueError(f'Pupil should be either 2D or 3D, not {pupil_array.ndim}')

    ncols = 2
    cmaps = ['inferno', 'twilight']
    col_titles = ['modulus', 'phase']
    figure, axes = plt.subplots(nrows, ncols, figsize=(ncols * _FIG_SIZE, nrows * _FIG_SIZE))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.T
    for (col_index, axis), pupil, cmap, title in zip(enumerate(axes), pupil_list, cmaps, col_titles):
        cbar_min = np.min(pupil)
        cbar_max = np.max(pupil)
        norm = plt.Normalize(cbar_min, cbar_max)
        for (row_index, ax), image, row_title in zip(enumerate(axis), pupil, row_titles):
            im = ax.imshow(apply_disk_mask(image), norm=norm, cmap=cmap)
            colorbar(im, cbar_ticks=[cbar_min, cbar_max])
            x_ticks = [0, image.shape[1]]
            xtick_labels = x_ticks
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(xtick_labels, fontsize=_TICK_SIZE)
            y_ticks = [0, image.shape[0]]
            ax.set_yticks(y_ticks)
            ytick_labels = y_ticks
            ax.set_yticklabels(ytick_labels, fontsize=_TICK_SIZE)
            if row_index == 0:
                ax.set_title(title, fontsize=_TITLE_SIZE)
            if nrows > 1 and col_index == 0:
                ax.text(-0.1, 0.5, row_title, fontsize=_TITLE_SIZE, verticalalignment='center',
                        rotation=90, transform=ax.transAxes)
                plt.subplots_adjust(left=0.05)

    plt.suptitle(f'Pupil properties ({name_of_propagator})', fontsize=_SUP_TITLE_SIZE)
    figure.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()


def plot_psf(
        psf: tp.Union[torch.Tensor, np.ndarray],
        name_of_propagator: str,
        quantity: str = 'modulus',
        z_slice_number: int = None,
        x_slice_number: int = None,
        y_slice_number: int = None,
        filepath: str = None
):
    """
    Plot the intensity or modulus or phase of a PSF, applicable to all four propagators.

    Parameters
    ----------
    psf : torch.Tensor or np.ndarray
        PSF image to plot.
    name_of_propagator : str
        Name of the propagator.
    quantity : str, optional
        Quantity of the PSF to plot. Default is 'modulus'. Valid choices are 'modulus', 'phase', 'intensity'.
    z_slice_number : int, optional
        Z slice number for the x-y plane.
    x_slice_number : int, optional
        X slice number for the y-z plane.
    y_slice_number : int, optional
        Y slice number for the x-z plane.
    filepath : str, optional
        Path to save the plot. Default is None, no file is saved.

    """
    # convert to numpy array
    psf_array = convert_tensor_to_array(psf)
    # check and compute quantity
    valid_choices = ['modulus', 'phase', 'intensity']
    if quantity == 'modulus':
        psf_quantity = np.abs(psf_array)
        cmap = 'inferno'
    elif quantity == 'phase':
        psf_quantity = np.angle(psf_array)
        cmap = 'twilight'
    elif quantity == 'intensity':
        psf_quantity = _compute_psf_intensity(psf_array)
        cmap = 'inferno'
    else:
        raise ValueError(f'quantity {quantity} is not supported, choose from {valid_choices}')

    number_of_pixel_z, dim, number_of_pixel_x, number_of_pixel_y = psf_quantity.shape
    if z_slice_number is None:
        z_slice_number = int(number_of_pixel_z // 2)
    if x_slice_number is None:
        x_slice_number = int(number_of_pixel_x // 2)
    if y_slice_number is None:
        y_slice_number = int(number_of_pixel_y // 2)

    psf_quantity = psf_quantity.swapaxes(0, 1)
    if dim == 1:
        row_titles = ['']
    elif dim == 3:
        row_titles = [r'$E_x$', r'$E_y$', r'$E_z$']
    else:
        raise ValueError(f'Number of channels of the PSF should be 1 or 3, not {dim}')
    psf_list = [
                   psf_quantity[:, z_slice_number, :, :],
                   psf_quantity[:, :, x_slice_number, :],
                   psf_quantity[:, :, :, y_slice_number]
                  ]

    nrows = dim
    ncols = len(psf_list)
    col_titles = [
                  f'x-y plane at z={z_slice_number}/{number_of_pixel_z} slices',
                  f'y-z plane at x={x_slice_number}/{number_of_pixel_x} slices',
                  f'x-z plane at y={y_slice_number}/{number_of_pixel_y} slices'
    ]
    cbar_min = min(np.min(psf) for psf in psf_list)
    cbar_max = max(np.max(psf) for psf in psf_list)
    norm = plt.Normalize(cbar_min, cbar_max)
    figure, axes = plt.subplots(nrows, ncols, figsize=(ncols * _FIG_SIZE, nrows * _FIG_SIZE))
    if dim == 1:
        axes = axes.reshape(1, -1)
    axes = axes.T
    for (col_index, axis), psf, col_title in zip(enumerate(axes), psf_list, col_titles):
        for (row_index, ax), image, row_title, in zip(enumerate(axis), psf, row_titles):
            im = ax.imshow(image, norm = norm, cmap=cmap)
            colorbar(im, cbar_ticks=[cbar_min, cbar_max])
            if row_index == 0:
                ax.set_title(col_title, fontsize=_TITLE_SIZE)
            if dim > 1 and col_index == 0:
                ax.text(-0.1, 0.5, row_title, fontsize=_TITLE_SIZE, verticalalignment='center',
                        rotation=90, transform=ax.transAxes)
                plt.subplots_adjust(left=0.05)
            x_ticks = [0, image.shape[1]]
            xtick_labels = x_ticks
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(xtick_labels, fontsize=_TICK_SIZE)
            y_ticks = [0, image.shape[0]]
            ax.set_yticks(y_ticks)
            ytick_labels = y_ticks
            ax.set_yticklabels(ytick_labels, fontsize=_TICK_SIZE)
    plt.suptitle(f'{quantity} of PSF at three orthogonal planes ({name_of_propagator})', fontsize=_SUP_TITLE_SIZE)
    figure.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()


def plot_runtime_benchmark_results(
        results: list,
        quantity: str,
        labels: list,
        title: str,
        filepath: str = None
):
    """
    Plot results of the runtime benchmarking.

    Parameters
    ----------
    results : list
        List that contains tuples of the variable to be benchmarked and the resulting value.
    quantity: str
        Quantity on the x-axis of the plot. Choose from 'pupil' or 'psf'.
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
        ax.loglog(x[1:], y[1:], label=label, ls=ls, marker='.', markersize=markersize, lw=lw, color=color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_ylabel('Time / s', fontsize=_LABEL_SIZE)

    ax.legend(fontsize=_TICK_SIZE)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.set_xlabel(f'Numerical size of the {quantity} / pixels', fontsize=_LABEL_SIZE)

    plt.grid(color='gray', ls='dotted', lw=lw)
    figure.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()


def plot_accuracy_benchmark_results(
        results: list,
        labels: list,
        title: str,
        filepath: str = None
):
    """
    Plot results of the accuracy benchmarking.

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
    -------

    """
    figure, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE * 1.5, _FIG_SIZE))
    colors = ['red', 'blue']
    for result, label, color in zip(results, labels, colors):
        x, y = zip(*result)
        ls = 'solid'
        ax.semilogx(x, y, base=2, label=label, ls=ls, marker='.', markersize=markersize, lw=lw, color=color)

    ax.legend(fontsize=_TICK_SIZE)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.set_xlabel('Numerical size of the pupil / pixels', fontsize=_LABEL_SIZE)
    ax.set_ylabel('Accuracy (MSE)', fontsize=_LABEL_SIZE)
    xs, _ = zip(*results[0])
    xticks = [entry - 1 for entry in xs]
    ax.set_xticks(xticks)
    xticklabels = xticks
    ax.set_xticklabels(xticks)

    plt.grid(color='gray', ls='dotted', lw=lw)
    figure.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()