import os
import sys

module_path = os.path.abspath(os.path.join('')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from propagator import ScalarCartesianPropagator, ScalarPolarPropagator
from pupil import ScalarCartesianPupil, ScalarPolarPupil

## Scalar benchmark

# Parameters
n_pix_psf = 201
na = 0.9
wavelength = 632
fov = 3000
defocus = 0
n_defocus = 1

# Loop over n_pix_pupil
n_points = 9
n_pix_pupils = np.logspace(5, 13, n_points, base=2)
cartesian_cpu_time = torch.zeros(n_points)
polar_cpu_time = torch.zeros(n_points)
cartesian_gpu_time = torch.zeros(n_points)
polar_gpu_time = torch.zeros(n_points)

# CPU timing benchmark
torch.cuda.synchronize()
n_rep = 10
device = "cpu"
for (i_pupil, n_pix_pupil) in tqdm(enumerate(n_pix_pupils)):
    n_pix_pupil = int(n_pix_pupil)

    # Cartesian CPU
    init_time = time()
    for i_rep in range(n_rep):
        pupil1 = ScalarCartesianPupil(n_pix_pupil, device=device)
        propagator1 = ScalarCartesianPropagator(pupil1, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na, fov=fov,
                                                defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device)
        field1 = propagator1.compute_focus_field()
    cartesian_cpu_time[i_pupil] = (time() - init_time) / n_rep

    # Polar CPU
    init_time = time()
    for i_rep in range(n_rep):
        pupil2 = ScalarPolarPupil(n_pix_pupil, device=device)
        propagator2 = ScalarPolarPropagator(pupil2, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na, fov=fov,
                                            defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device)
        field2 = propagator2.compute_focus_field()
    polar_cpu_time[i_pupil] = (time() - init_time) / n_rep

    # Cartesian GPU
    device = "cuda:0"
    torch.cuda.synchronize()
    init_time = time()
    for i_rep in range(n_rep):
        pupil1 = ScalarCartesianPupil(n_pix_pupil, device=device)
        propagator1 = ScalarCartesianPropagator(pupil1, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na, fov=fov,
                                                defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device)
        field1 = propagator1.compute_focus_field()
        torch.cuda.synchronize()
    cartesian_gpu_time[i_pupil] = (time() - init_time) / n_rep

    # Polar GPU
    torch.cuda.synchronize()
    init_time = time()
    for i_rep in range(n_rep):
        pupil2 = ScalarPolarPupil(n_pix_pupil, device=device)
        propagator2 = ScalarPolarPropagator(pupil2, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na, fov=fov,
                                            defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device)
        field2 = propagator2.compute_focus_field()
        torch.cuda.synchronize()
    polar_gpu_time[i_pupil] = (time() - init_time) / n_rep

# Save
# np.save('data/cpu_time_cartesian.npy', cartesian_cpu_time.cpu().numpy())
# np.save('data/cpu_time_polar.npy', polar_cpu_time.cpu().numpy())
np.save('data/gpu_time_cartesian.npy', cartesian_gpu_time.cpu().numpy())
np.save('data/gpu_time_polar.npy', polar_gpu_time.cpu().numpy())
