"""
A collection of plotting functions.

"""
import os
import typing as tp
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable

from .misc import convert_tensor_to_array

_FIG_SIZE = 5
_SUP_TITLE_SIZE = 17
_TITLE_SIZE = 12
_LABEL_SIZE = 18
_TICK_SIZE = 16
lw = 1
markersize = 6


def colorbar(mappable, cbar_ticks: tp.Union[str, tp.List, None] = 'auto', tick_size: float = _TICK_SIZE,
             cbar_labels: tp.List[str] = None):
    """
    Colorbar with the option to add or remove ticks.

    Parameters
    ----------
    mappable :
        Matplotlib Mappable.
    cbar_ticks : None or str or List of ticks
        If None, no ticks visible. If 'auto': ticks are determined automatically. Otherwise, set the ticks as given by cbar_ticks.
    tick_size: float, optional
        Fontsize of the tick labels.
    cbar_labels: list[str], optional
        Cbar labels. Default is None, use cbar ticks.

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
        if cbar_labels is not None:
            if len(cbar_labels) != len(cbar_ticks):
                raise ValueError('The length of the cbar labels and ticks are different.')
            else:
                cbar.set_ticklabels(cbar_labels, fontsize=tick_size)
        else:
            cbar.set_ticklabels([f'{tick:.2f}' for tick in cbar_ticks], fontsize=tick_size)
    plt.sca(last_axes)
    return cbar


def apply_disk_mask(img):
    """Apply a disk mask to a square image."""
    img = img.copy()
    lx, ly = img.shape
    diameter = max(lx, ly)
    # check if square
    if lx != ly:
        msg = f'Image is non-square, shape: {img.shape}. Applying an over-sized disk mask!'
        warnings.warn(msg)
    # create mask
    mask = np.zeros((lx, ly))
    i = np.linspace(0, lx, lx)
    j = np.linspace(0, ly, ly)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    disk = (ii - lx // 2) ** 2 + (jj - ly // 2) ** 2 <= (diameter // 2) ** 2
    mask[disk] = 1
    # apply mask, set values outside the mask to nan
    img = np.where(mask, img, np.nan)
    return img


def _compute_psf_intensity(input_image: np.ndarray) -> np.ndarray:
    r"""
    Compute the intensity of a complex field.

    The input array must be 4D with this convention:

    - dim one: z axis, or defocus slices.

    - dim two: electric field components. Only one for scalar and three :math:`(\mathbf{e}_x, \mathbf{e}_y, \mathbf{e}_z)` for vectorial.

    - dim three and four: :math:`(x, y)` axes.

    The intensity is computed as follows:

    .. math:: I = \sum_{i=1}^{N} |\mathbf{e}_i(x, y, z)|^2, \quad N = 1 \, \mathrm{or} \, 3.

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
        filepath: str = None,
        show_cbar_ticks: bool = False,
        show_image_ticks: bool = False,
        show_titles: bool = True,
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
    show_titles : bool, optional
        Whether to show the titles on the first row. Default is False.
    show_image_ticks : bool, optional
        Whether to show ticks. Default is False.
    show_cbar_ticks : bool, optional
        Whether to show the ticks for the colorbar. Default is False.

    """
    if 'spherical' in name_of_propagator:
        raise NotImplementedError('For spherical propagators, the pupil is represented by two 1D intervals, '
                                  'no 2D image is thus available. '
                                  'Please check the pupil of the equivalent Cartesian propagator instead.')
    # convert to numpy array
    pupil_array = convert_tensor_to_array(pupil).squeeze()
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
        row_titles = [r'$\mathbf{e}_x$', r'$\mathbf{e}_y$', r'$\mathbf{e}_z$']
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
        if show_cbar_ticks:
            cbar_ticks = [cbar_min, cbar_max]
        else:
            cbar_ticks = None
        for (row_index, ax), image, row_title in zip(enumerate(axis), pupil, row_titles):
            im = ax.imshow(apply_disk_mask(image), norm=norm, cmap=cmap)
            colorbar(im, cbar_ticks=cbar_ticks)
            if show_image_ticks:
                x_ticks = [0, image.shape[1]]
                xtick_labels = x_ticks
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(xtick_labels, fontsize=_TICK_SIZE)
                y_ticks = [0, image.shape[0]]
                ax.set_yticks(y_ticks)
                ytick_labels = y_ticks
                ax.set_yticklabels(ytick_labels, fontsize=_TICK_SIZE)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            if show_titles and row_index == 0:
                ax.set_title(title, fontsize=_TITLE_SIZE)
            if nrows > 1 and col_index == 0:
                ax.text(-0.1, 0.5, row_title, fontsize=_TITLE_SIZE, verticalalignment='center',
                        rotation=90, transform=ax.transAxes)
                plt.subplots_adjust(left=0.05)

    plt.suptitle(f'Pupil properties ({name_of_propagator})', fontsize=_SUP_TITLE_SIZE)

    if filepath is not None:
        figure.tight_layout()
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
        filepath: str = None,
        show_cbar_ticks: bool = False,
        show_image_ticks: bool = False,
        show_titles: bool = False,

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
        Quantity of the PSF to plot. Default is 'modulus'. Valid choices are 'modulus', 'phase', 'intensity', 'amplitude'.
    z_slice_number : int, optional
        Z slice number for the x-y plane.
    x_slice_number : int, optional
        X slice number for the y-z plane.
    y_slice_number : int, optional
        Y slice number for the x-z plane.
    filepath : str, optional
        Path to save the plot. Default is None, no file is saved.
    show_titles : bool, optional
        Whether to show the titles on the first row. Default is False.
    show_image_ticks : bool, optional
        Whether to show ticks. Default is False.
    show_cbar_ticks : bool, optional
        Whether to show the ticks for the colorbar. Default is False.

    """
    # convert to numpy array
    psf_array = convert_tensor_to_array(psf)
    # check and compute quantity
    valid_choices = ['modulus', 'phase', 'intensity', 'amplitude']
    if quantity == 'modulus':
        psf_quantity = np.abs(psf_array)
        cmap = 'inferno'
    elif quantity == 'phase':
        psf_quantity = np.angle(psf_array)
        cmap = 'twilight'
    elif quantity == 'intensity':
        psf_quantity = _compute_psf_intensity(psf_array)
        cmap = 'inferno'
    elif quantity == 'amplitude':
        psf_quantity = np.sqrt(_compute_psf_intensity(psf_array))
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
        row_titles = [r'$\mathbf{e}_x$', r'$\mathbf{e}_y$', r'$\mathbf{e}_z$']
    else:
        raise ValueError(f'Number of channels of the PSF should be 1 or 3, not {dim}')

    if number_of_pixel_z == 1: # 2D PSF
        psf_slice = psf_quantity[:, 0, :, :]

        cbar_min = np.min(psf_slice)
        cbar_max = np.max(psf_slice)
        norm = plt.Normalize(cbar_min, cbar_max)

        figure, axes = plt.subplots(dim, 1, figsize=(1 * _FIG_SIZE, dim * _FIG_SIZE))

        if dim == 1:
            axes = [axes]

        for row_index, (ax, image, row_title) in enumerate(zip(axes, psf_slice, row_titles)):
            im = ax.imshow(image, norm=norm, cmap=cmap)
            colorbar(im, cbar_ticks=[cbar_min, cbar_max] if show_cbar_ticks else None)
            if show_titles:
                ax.set_title('XY-plane (2D PSF)', fontsize=_TITLE_SIZE)
            if dim > 1 :
                ax.set_ylabel(row_title, fontsize=_LABEL_SIZE)
                plt.subplots_adjust(left=0.05)
            if show_image_ticks:
                x_ticks = [0, image.shape[1]]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticks, fontsize=_TICK_SIZE)
                y_ticks = [0, image.shape[0]]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_ticks, fontsize=_TICK_SIZE)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.suptitle(f'{quantity.capitalize()} ({name_of_propagator.capitalize()})', fontsize=_SUP_TITLE_SIZE)
    else: # 3D PSF
        psf_list = [
                       psf_quantity[:, z_slice_number, :, :],
                       psf_quantity[:, :, x_slice_number, :],
                       psf_quantity[:, :, :, y_slice_number],
                      ]

        nrows = dim
        ncols = len(psf_list)
        col_titles = [
                      f'XY-plane (z={z_slice_number+1}/{number_of_pixel_z} slice)',
                      f'ZY plane (x={x_slice_number+1}/{number_of_pixel_x} slice)',
                      f'ZX plane (y={y_slice_number+1}/{number_of_pixel_y} slice)',
        ]
        cbar_min = min(np.min(psf) for psf in psf_list)
        cbar_max = max(np.max(psf) for psf in psf_list)
        norm = plt.Normalize(cbar_min, cbar_max)
        if show_cbar_ticks:
            cbar_ticks = [cbar_min, cbar_max]
        else:
            cbar_ticks = None
        figure, axes = plt.subplots(nrows, ncols, figsize=(ncols * _FIG_SIZE, nrows * _FIG_SIZE))
        if dim == 1:
            axes = axes.reshape(1, -1)
        axes = axes.T
        for (col_index, axis), psf, col_title in zip(enumerate(axes), psf_list, col_titles):
            for (row_index, ax), image, row_title, in zip(enumerate(axis), psf, row_titles):
                im = ax.imshow(image, norm = norm, cmap=cmap)
                colorbar(im, cbar_ticks=cbar_ticks)
                if show_titles and row_index == 0:
                    ax.set_title(col_title, fontsize=_TITLE_SIZE)
                if dim > 1 and col_index == 0:
                    ax.set_ylabel(row_title, fontsize=_LABEL_SIZE)
                    plt.subplots_adjust(left=0.05)
                if show_image_ticks:
                    x_ticks = [0, image.shape[1]]
                    xtick_labels = x_ticks
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(xtick_labels, fontsize=_TICK_SIZE)
                    y_ticks = [0, image.shape[0]]
                    ax.set_yticks(y_ticks)
                    ytick_labels = y_ticks
                    ax.set_yticklabels(ytick_labels, fontsize=_TICK_SIZE)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
        plt.suptitle(f'{quantity.capitalize()} at three orthogonal planes ({name_of_propagator.capitalize()})', fontsize=_SUP_TITLE_SIZE)
    if filepath is not None:
        figure.tight_layout()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()
