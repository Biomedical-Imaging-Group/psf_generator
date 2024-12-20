import json
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from psf_generator.utils.handle_data import load_from_npy

_FIG_SIZE = 8
_FONT_SIZE = 30

def plot_amplitude(img, base_name, dx: float, v_range:list = None, show_scale_bar=False):
    fig, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE, _FIG_SIZE))
    ax.set_xticks([])
    ax.set_yticks([])
    if v_range is not None:
        norm = plt.Normalize(v_range[0], v_range[1])
        ax.imshow(img, cmap='inferno', norm=norm)
    else:
        ax.imshow(img, cmap='inferno')
    if show_scale_bar:
        scalebar = ScaleBar(dx=dx, units="nm", pad=0.8,
                            frameon=False, color='w', fixed_value=1000,
                            font_properties={'size': _FONT_SIZE})
        ax.add_artist(scalebar)
    fig.tight_layout()
    save_path = os.path.join(base_plot_path, base_name + '.svg')
    fig.savefig(save_path, format='svg')

def plot_profiles(imgs, x_slice, base_name):
    labels = ['scalar', 'vectorial']
    fig, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE, _FIG_SIZE))
    for img, label in zip(imgs, labels):
        ax.plot(img[x_slice, :], label=label)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.legend(fontsize=_FONT_SIZE)
    fig.tight_layout()
    save_path = os.path.join(base_plot_path, base_name + '.svg')
    fig.savefig(save_path, format='svg')

def plot1():
    z, y, x = 128, 128, 128
    imgs = {}
    vmins, vmaxs = [], []
    for name in psf_names:
        data_path = os.path.join(base_data_path, name + '_psf.npy')
        data = load_from_npy(data_path)
        amplitude = np.sqrt(np.sum(np.abs(data) ** 2, axis=1))
        imgs[name, 'z'] = amplitude[z, :, :]
        imgs[name, 'y'] = amplitude[:, :, y]
        vmins.append(min(np.min(imgs[name, 'z']), np.min(imgs[name, 'y'])))
        vmaxs.append(max(np.max(imgs[name, 'z']), np.max(imgs[name, 'y'])))
    # take the min and max of both xy and xz planes for both propagators
    vmin = min(vmins)
    vmax = max(vmaxs)

    for axis in ('z', 'y'):
        plot_profiles([imgs[name, axis] for name in psf_names], x, f'{axis}_line_profile')
        for name in psf_names:
            json_path = os.path.join(base_data_path, name + '_params.json')
            with open(json_path) as file:
                params = json.load(file)
            dx = params['fov'] / params['n_pix_psf']
            if axis == 'z' and name == 'scalar_cartesian':
                show_scale_bar = True
            else:
                show_scale_bar = False
            plot_amplitude(img=imgs[name, axis], base_name=f'{name}_{axis}',dx=dx, v_range=[vmin, vmax], show_scale_bar=show_scale_bar)


def plot2():
    zs = [0, 128, -1]
    imgs = {}
    vmins, vmaxs = [], []
    for name in psf_names:
        data_path = os.path.join(base_data_path, name + '_psf.npy')
        data = load_from_npy(data_path)
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
            plot_amplitude(img=imgs[name, z], base_name=f'{name}_z{z}', dx=dx, v_range=[vmin, vmax])


if __name__ == "__main__":
    # define base file path
    exp_name = 'pure_gaussian_low_na_e0_1_0'
    base_plot_path = os.path.join('results', 'plots', 'fields', exp_name)
    os.makedirs(base_plot_path, exist_ok=True)

    base_data_path = os.path.join('results', 'data', 'fields', exp_name)
    psf_names = ['scalar_cartesian', 'vectorial_cartesian']
    plot1()

