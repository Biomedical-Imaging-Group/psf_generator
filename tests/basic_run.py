import torch
import matplotlib.pyplot as plt

from propagators import *

def plot_scalar_field(field):
    idx = n_defocus // 2
    ax_idx = 0
    print(field.shape)
    plt.figure()
    plt.subplot(121)
    plt.imshow(torch.abs(field[idx, ax_idx, ...].squeeze()))
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(torch.abs(field[:, ax_idx, n_pix_psf // 2, :].squeeze().T))
    plt.colorbar()
    plt.show()

def plot_vectorial_field(field):
    intensity = torch.sqrt(torch.sum(torch.abs(field[:, :, :, :].squeeze()) ** 2, dim=1))
    idx = n_defocus // 2
    print(field.shape)
    plt.figure()
    plt.subplot(121)
    plt.imshow(intensity[idx, ...].squeeze())
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(intensity[:, n_pix_psf // 2, :].squeeze().T)
    plt.colorbar()
    plt.show()

n_pix_pupil = 203
n_pix_psf = 201
na = 1.4
wavelength = 632
fov = 2000
defocus = 4000
n_defocus = 200
e0x = 1
e0y = 0

kwargs = {
    'n_pix_pupil': n_pix_pupil,
    'n_pix_psf': n_pix_psf,
    'wavelength': wavelength,
    'na': na,
    'fov': fov,
    'defocus_min': -defocus,
    'defocus_max': defocus,
    'n_defocus': n_defocus,
    'apod_factor': False,
    'gibson_lanni': True
}

# define propagators
propagators = [
    ScalarCartesianPropagator(**kwargs),
    ScalarPolarPropagator(**kwargs),
    VectorialCartesianPropagator(e0x=e0x, e0y=e0y, **kwargs),
    VectorialPolarPropagator(e0x=e0x, e0y=e0y, **kwargs)
]

for propagator in propagators:
    class_name = propagator.__class__.__name__
    print(class_name)
    field = propagator.compute_focus_field()
    if 'Scalar' in class_name:
        plot_scalar_field(field)
    if 'Vectorial' in class_name:
        plot_vectorial_field(field)

