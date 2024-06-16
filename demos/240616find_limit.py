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
n_pix_pupil = 8192
n_pix_psf = 128
NA = 0.9
wavelength = 632
fov = 3000
defocus = 0
n_defocus = 1
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # we want highest dimension, even if it's slow

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

result = (torch.abs(field1.squeeze())**2 + torch.abs(field2.squeeze())**2) / 2

# Save result