"""
Interferometric scattering microscopy (iSCAT) of a gold nanoparticle.

A 30 nm gold particle in water on a glass coverslip, imaged with an oil objective focused 1 um above the
coverslip (the configuration of Dong et al. 2021, J. Phys. D 54, 394002). The script computes the
interferometric PSF (contrast image) as a function of the height of the particle for iSCAT, COBRI and
dark-field, and saves three figures under results/plots/iscat/:

- images.png: the images recorded by the three techniques for a few particle heights;
- axial.png: the on-axis contrast, the amplitude and the phase of the scattered light versus the height;
- meridional.png: the meridional (x-z) section of the iSCAT iPSF.

"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from psf_generator.modalities import COBRIMicroscope, DarkFieldMicroscope, ISCATMicroscope, Particle

if __name__ == '__main__':
    plot_path = os.path.join('results', 'plots', 'iscat')
    os.makedirs(plot_path, exist_ok=True)

    # 30 nm gold at 517.5 nm (permittivity from Johnson & Christy), water / glass / oil, NA 1.3.
    gold = Particle(radius=15.0, permittivity=-3.7328 + 2.7725j)
    setup = dict(wavelength=517.5, na=1.3, n_s=1.33, n_g=1.5, n_i=1.5, z_focus=1000.0,
                 n_pix_pupil=201, n_pix_psf=101, pix_size=40.0)
    microscopes = {
        'iSCAT': ISCATMicroscope(gold, **setup),
        'COBRI': COBRIMicroscope(gold, **setup),
        'dark-field': DarkFieldMicroscope(gold, **setup),
    }
    iscat = microscopes['iSCAT']
    x = iscat.x.numpy() / 1e3  # um

    # ---- Images for a few heights of the particle ---------------------------------------------------------
    heights = [0.0, 780.0, 840.0, 890.0, 1000.0]
    positions = [(0.0, 0.0, z) for z in heights]
    fig, axes = plt.subplots(len(heights), 3, figsize=(7, 2.2 * len(heights)), constrained_layout=True)
    for column, (name, microscope) in enumerate(microscopes.items()):
        images = microscope.compute_image(positions).numpy()
        limit = np.abs(images - (microscope.reference_intensity if name != 'dark-field' else 0)).max()
        for row, image in enumerate(images):
            ax = axes[row, column]
            if name == 'dark-field':
                ax.imshow(image, extent=[x[0], x[-1], x[-1], x[0]], cmap='gray', vmin=0, vmax=limit)
            else:
                ax.imshow(image, extent=[x[0], x[-1], x[-1], x[0]], cmap='gray',
                          vmin=microscope.reference_intensity - limit, vmax=microscope.reference_intensity + limit)
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if column == 0:
                ax.set_ylabel(f'$z_p$ = {heights[row] / 1e3:.2f} um')
    fig.suptitle('Detected intensity (4 x 4 um field of view, focus at 1 um)')
    fig.savefig(os.path.join(plot_path, 'images.png'), dpi=150)

    # ---- On-axis signal versus the height of the particle -------------------------------------------------
    z = torch.linspace(0.0, 3000.0, 601)
    positions = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1)
    centre = setup['n_pix_psf'] // 2
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True, constrained_layout=True)
    for name, microscope in microscopes.items():
        reference, scattered = microscope.compute_fields(positions)
        on_axis = scattered[:, 0, centre, centre]
        if name != 'dark-field':
            contrast = microscope.compute_contrast(positions)[:, centre, centre]
            axes[0].plot(z / 1e3, contrast, label=name)
            phase = torch.angle(on_axis * reference[0].conj())
            axes[2].plot(z / 1e3, phase, label=name)
        axes[1].plot(z / 1e3, on_axis.abs(), label=name)
    axes[0].set_ylabel('on-axis contrast'), axes[0].legend()
    axes[1].set_ylabel('|E_sca| / |E_0|')
    axes[2].set_ylabel('phase (rad)'), axes[2].set_xlabel('height of the particle $z_p$ (um)')
    fig.savefig(os.path.join(plot_path, 'axial.png'), dpi=150)

    # ---- Meridional section of the iSCAT iPSF -------------------------------------------------------------
    z = torch.linspace(0.0, 2000.0, 201)
    positions = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1)
    contrast = iscat.compute_contrast(positions).numpy()
    section = contrast[:, centre, :]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    limit = np.abs(section).max()
    image = ax.imshow(section, extent=[x[0], x[-1], z[-1] / 1e3, z[0] / 1e3], cmap='RdBu_r', vmin=-limit, vmax=limit,
                      aspect='auto', origin='upper')
    ax.invert_yaxis()
    ax.set_xlabel('x (um)'), ax.set_ylabel('height of the particle $z_p$ (um)')
    ax.set_title('iSCAT contrast, meridional section (focus at 1 um)')
    fig.colorbar(image, ax=ax)
    fig.savefig(os.path.join(plot_path, 'meridional.png'), dpi=150)
    print(f'Figures saved under {plot_path}')
