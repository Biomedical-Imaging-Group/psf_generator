import os
import sys

module_path = os.path.abspath(os.path.join('')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)
import numpy as np
import torch
from tqdm import tqdm

module_path = os.path.abspath(os.path.join('..')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from propagator import ScalarCartesianPropagator, ScalarPolarPropagator
from pupil import ScalarCartesianPupil, ScalarPolarPupil

from torch.special import bessel_j1

## Scalar benchmark

# Parameters
n_pix_psf = 201
na = 0.9
wavelength = 632
fov = 3000
defocus = 0
n_defocus = 1
refractive_index = 1.5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Define the limit
airy_fun = lambda x: torch.where(x > 1e-6, 
                                 2 * bessel_j1(x) / x, 
                                 1 - x ** 2 / 8)
x = torch.linspace(-fov/2, fov/2, n_pix_psf)
xx, yy = torch.meshgrid(x, x, indexing='ij')
rr = torch.sqrt(xx ** 2 + yy ** 2)
k = 2 * np.pi / wavelength
limit = airy_fun(k * rr * na / refractive_index)

# Loop over n_pix_pupil
start_power = 3
n_points = 8
n_pix_pupils = np.logspace(start_power, start_power+n_points-1, n_points, base=2) + 1
cartesian_err = torch.zeros(n_points)
polar_err = torch.zeros(n_points)

for (i_pupil, n_pix_pupil) in tqdm(enumerate(n_pix_pupils)):
    n_pix_pupil = int(n_pix_pupil)

    # Cartesian
    pupil1 = ScalarCartesianPupil(n_pix_pupil, device=device)
    propagator1 = ScalarCartesianPropagator(pupil1, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na, fov=fov,
                                            defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device,
                                            sz_correction=False, apod_factor=False)
    field1 = propagator1.compute_focus_field()
    field1 = field1 / torch.max(torch.abs(field1))

    # Polar
    pupil2 = ScalarPolarPupil(n_pix_pupil, device=device)
    propagator2 = ScalarPolarPropagator(pupil2, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na, fov=fov,
                                        defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device,
                                        cos_factor=True, apod_factor=False)
    field2 = propagator2.compute_focus_field()
    field2 = field2 / torch.max(torch.abs(field2))

    cartesian_err[i_pupil] = np.sqrt(torch.sum(torch.abs(field1.cpu().squeeze() - limit)**2))
    polar_err[i_pupil] = np.sqrt(torch.sum(torch.abs(field2.cpu().squeeze() - limit)**2))

# Save
np.save('benchmark_cartesian.npy', cartesian_err.cpu().numpy())
np.save('benchmark_polar.npy', polar_err.cpu().numpy())
