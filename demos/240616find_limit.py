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


# Parameters
n_pix_pupil = 65536
n_pix_psf = 201
na = 0.9
wavelength = 632
fov = 3000
defocus = 0
n_defocus = 1
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # we want highest dimension, even if it's slow

# Cartesian
print("Computing Cartesian model")
pupil1 = ScalarCartesianPupil(n_pix_pupil, device=device)
propagator1 = ScalarCartesianPropagator(pupil1, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na, fov=fov,
                                        defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device)
field1 = propagator1.compute_focus_field()

intensity1 = torch.abs(field1.squeeze())**2
intensity1 = intensity1.cpu().numpy()
np.save(f'data/limit{n_pix_pupil}_carte.npy', intensity1)
print("Cartesian model saved.")

# Polar
print("Computing Polar model")
pupil2 = ScalarPolarPupil(n_pix_pupil, device=device)
propagator2 = ScalarPolarPropagator(pupil2, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na, fov=fov,
                                    defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device)
field2 = propagator2.compute_focus_field()

intensity2 = torch.abs(field2.squeeze())**2
intensity2 = intensity2.cpu().numpy()
np.save(f'data/limit{n_pix_pupil}_polar.npy', intensity2)
print("Polar model saved.")
