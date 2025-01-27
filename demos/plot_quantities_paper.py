import json
import os

import numpy as np
from matplotlib import pyplot as plt

from matplotlib_scalebar.scalebar import ScaleBar
from psf_generator.utils.handle_data import load_from_npy
from psf_generator.utils.plots import apply_disk_mask, colorbar

_FIG_SIZE = 8
_BAR_SIZE = 40
_FONT_SIZE = 42
lw = 2

def plot_amplitude(img: np.ndarray, base_name: str, units: list, bar_value: float, v_range:list = None):
    """
    Plot amplitude of the PSF.

    Parameters
    ----------
    img : np.ndarray
        Image to plot.
    base_name : str
        name of the base path to the image data.
    units : list
        Pixel sizes in the both directions of the image.
    bar_value : float
        Physical length of the scale bars.
    v_range : list, optional
        Min and Max values of the colorbar. Default is None, in which case the image is not normalized.

    """
    dx, dz = units
    fig, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE, _FIG_SIZE))

    ax.set_xticks([])
    ax.set_yticks([])

    if v_range is not None:
        norm = plt.Normalize(v_range[0], v_range[1])
        im = ax.imshow(img, cmap='inferno', norm=norm)
    else:
        im = ax.imshow(img, cmap='inferno')
    colorbar(im, cbar_ticks=None)
    scalebar = ScaleBar(dx=dx, units="nm", pad=0.8, scale_loc='none',
                        frameon=False, color='w', fixed_value=bar_value,
                        rotation='horizontal',
                        font_properties={'size': _BAR_SIZE})
    ax.add_artist(scalebar)
    if dz is not None:
        scalebar2 = ScaleBar(dx=dz, units='nm', pad=0.8, scale_loc='none',
                             frameon=False, color='w', fixed_value=bar_value,
                             rotation='vertical',
                             font_properties={'size': _BAR_SIZE})
        ax.add_artist(scalebar2)
    fig.tight_layout()
    save_path = os.path.join(base_plot_path, base_name + '.svg')
    fig.savefig(save_path, format='svg')

def plot_profiles(imgs: list, base_name: str, phy_x: float, v_range: list):
    """
    Plot line profiles of the images.

    Parameters
    ----------
    imgs : list
        List of images to extract line profiles from.
    base_name : str
        name of the base path to the image data.
    phy_x : float
        Physical length of the axis along which the line is drawn.
    v_range : list
        Min and Max values of the colorbar. Default is None, in which case the image is not normalized.

    """
    labels = ['scalar', 'vectorial']
    fig, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE*1.1, _FIG_SIZE))
    for img, label in zip(imgs, labels):
        x_size = img.shape[1]
        ax.plot(img[:, img.shape[0] // 2], label=label, lw=lw)
        ax.set_xticks([0, x_size // 2, x_size])
        ax.set_xticklabels([- phy_x // 2, 0, phy_x // 2], fontsize=_FONT_SIZE)
        ax.set_yticks(v_range)
        ax.set_yticklabels([0, v_range[1]], fontsize=_FONT_SIZE)
        ax.axhline(ls='dashed', lw=lw, color='gray')
    ax.legend(fontsize=_FONT_SIZE)
    fig.tight_layout()
    save_path = os.path.join(base_plot_path, base_name + '.svg')
    fig.savefig(save_path, format='svg')


def plot_phase(phase: np.ndarray, base_name: str, normalize: bool):
    """
    Plot the phase image.

    Parameters
    ----------
    phase : np.ndarray
        Phase image to plot.
    base_name : str
        name of the base path to the image data.
    normalize : bool
        Whether to normalize the value of the image.

    """
    fig, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE, _FIG_SIZE))
    if normalize:
        cbar_min, cbar_max = -np.pi, np.pi
        cbar_labels = [r'-$\pi$', r'$\pi$']
        cmap = 'twilight'
    else:
        cbar_min, cbar_max = np.nanmin(phase), np.nanmax(phase)
        cbar_labels = [np.round(cbar_min, 1), np.round(cbar_max, 1)]
        cmap = 'viridis'
    norm = plt.Normalize(cbar_min, cbar_max)
    im = ax.imshow(phase, norm=norm, cmap=cmap)

    colorbar(im, cbar_ticks=[cbar_min, cbar_max], cbar_labels=cbar_labels, tick_size=_FONT_SIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    save_path = os.path.join(base_plot_path, base_name + '.svg')
    fig.savefig(save_path, format='svg')

def plot_vec_phases(normalize: bool = True):
    """
    Plot the phases of a given pupil image.

    Parameters
    ----------
    normalize: bool, optional
        Whether to normalize the value of the image. Default is True.

    """
    for name in psf_names:
        data_path = os.path.join(base_data_path, name + '_pupil.npy')
        data = np.squeeze(load_from_npy(data_path))
        phases = np.angle(data)
        for (number, phase) in enumerate(phases):
            phase = apply_disk_mask(phase)
            plot_phase(phase, base_name=f'{name}_pupil_{number}', normalize=normalize)


def plot1(plot_profile: bool = True):
    """
    PLot the amplitude and the line profiles of given PSFs.

    Parameters
    ----------
    plot_profile: bool, optional
        Whether to plot the line profile figure. Default is True.

    """
    imgs = {}
    vmins, vmaxs = [], []
    for name in psf_names:
        data_path = os.path.join(base_data_path, name + '_psf.npy')
        data = load_from_npy(data_path)
        amplitude = np.sqrt(np.sum(np.abs(data) ** 2, axis=1))
        z = data.shape[0] // 2
        y = data.shape[2] // 2
        imgs[name, 'z'] = amplitude[z, :, :]
        imgs[name, 'y'] = amplitude[:, :, y]
        print(f'energy of xy slice for {name}: {np.sum(imgs[name, "z"])}')
        print(f'energy of xz slice for {name}: {np.sum(imgs[name, "y"])}')
        vmins.append(min(np.min(imgs[name, 'z']), np.min(imgs[name, 'y'])))
        vmaxs.append(max(np.max(imgs[name, 'z']), np.max(imgs[name, 'y'])))
    # take the min and max of both xy and xz planes for both propagators
    vmin = np.round(min(vmins), 2)
    vmax = np.round(max(vmaxs), 2)

    for axis in ('z', 'y'):
        for name in psf_names:
            json_path = os.path.join(base_data_path, name + '_params.json')
            with open(json_path) as file:
                params = json.load(file)
            dx = params['fov'] / params['n_pix_psf']
            if axis == 'z':
                phy_size = [params['fov'], params['fov']]
                units = [dx, None]
            else:
                phy_z = params['defocus_max'] - params['defocus_min']
                dz = phy_z / params['n_defocus']
                phy_size = [phy_z, params['fov']]
                units = [dx, dz]
            plot_amplitude(img=imgs[name, axis], base_name=f'{name}_{axis}', units=units,  bar_value=bar_value,
                           v_range=[vmin, vmax])
            if plot_profile:
                plot_profiles([imgs[name, axis] for name in psf_names], base_name=f'{axis}_line_profile',
                            phy_x=int(phy_size[0] / 1e3), v_range=[vmin, vmax])


def plot2():
    """
    Plot the xy planes at 3 z-positions of a given PSF.
    Useful for showing astigmatism.

    """
    imgs = {}
    vmins, vmaxs = [], []
    for name in psf_names:
        data_path = os.path.join(base_data_path, name + '_psf.npy')
        data = load_from_npy(data_path)
        zs = [0, data.shape[0]//2, data.shape[0]-1]
        amplitude = np.sqrt(np.sum(np.abs(data) ** 2, axis=1))
        for z in zs:
            imgs[name, z] = amplitude[z, :, :]
            vmins.append(np.min(imgs[name, z]))
            vmaxs.append(np.max(imgs[name, z]))
    vmin = min(vmins)
    vmax = max(vmaxs)

    for z in zs:
        for name in psf_names:
            json_path = os.path.join(base_data_path, name + '_params.json')
            with open(json_path) as file:
                params = json.load(file)
            dx = params['fov'] / params['n_pix_psf']
            units = [dx, None]
            plot_amplitude(img=imgs[name, z], base_name=f'{name}_z{z}', units=units, bar_value=bar_value, v_range=[vmin, vmax])


if __name__ == "__main__":
    # define base file path
    exp_name = 'gl_gaussian_high_na_e0_1_1j'
    bar_value = 3000 if 'low_na' in exp_name else 600
    base_plot_path = os.path.join('results', 'plots', 'fields', exp_name)
    os.makedirs(base_plot_path, exist_ok=True)

    base_data_path = os.path.join('results', 'data', 'fields', exp_name)
    psf_names = ['vectorial_cartesian']
    # plot2()
    plot1(plot_profile=False)
    plot_vec_phases(normalize=False)


