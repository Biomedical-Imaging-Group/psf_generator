import os
import sys
module_path = os.path.abspath(os.path.join('')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)
import numpy as np
from matplotlib import pyplot as plt

import torch
from pupil import ScalarCartesianPupil, ScalarPolarPupil
from propagator import ScalarCartesianPropagator, ScalarPolarPropagator
from tqdm import tqdm

## Scalar benchmark

# Parameters
n_pix_psf = 201
NA = 0.9
wavelength = 632
fov = 3000
defocus = 0
n_defocus = 1
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load the limit
limit_size = 32768
cartesian_psf = np.load(f'data/limit{limit_size}_carte.npy')
polar_psf = np.load(f'data/limit{limit_size}_polar.npy')
limit = (cartesian_psf + polar_psf) / 2

# Loop over n_pix_pupil
n_points = 9
n_pix_pupils = np.logspace(5, 13, n_points, base=2)
cartesian_err = torch.zeros(n_points)
polar_err = torch.zeros(n_points)

for (i_pupil, n_pix_pupil) in tqdm(enumerate(n_pix_pupils)):
    n_pix_pupil = int(n_pix_pupil)

    # Cartesian
    pupil1 = ScalarCartesianPupil(n_pix_pupil, device=device)
    propagator1 = ScalarCartesianPropagator(pupil1, n_pix_psf=n_pix_psf, wavelength=wavelength, NA=NA, fov=fov,
                                            defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device)
    field1 = propagator1.compute_focus_field()

    # Polar
    pupil2 = ScalarPolarPupil(n_pix_pupil, device=device)
    propagator2 = ScalarPolarPropagator(pupil2, n_pix_psf=n_pix_psf, wavelength=wavelength, NA=NA, fov=fov,
                                        defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device)
    field2 = propagator2.compute_focus_field()

    cartesian_err[i_pupil] = torch.sum((torch.abs(field1.cpu().squeeze())**2 - limit)**2)
    polar_err[i_pupil] = torch.sum((torch.abs(field2.cpu().squeeze())**2 - limit)**2)

# Save
np.save('data/benchmark_cartesian.npy', cartesian_err.cpu().numpy())
np.save('data/benchmark_polar.npy', polar_err.cpu().numpy())
