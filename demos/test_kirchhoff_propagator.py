import os
import sys

import matplotlib.pyplot as plt
import numpy as np

module_path = os.path.abspath(os.path.join('..')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from params import Params
from pupil import ScalarPupil
from propagator import KirchhoffPropagator

FIGSIZE = (8, 8)

user_input = {
    'wavelength': 600,
    'NA': 1.1,
    'refractive_index': None
}

params = Params(user_input)
pupil = ScalarPupil(params)
propagator = KirchhoffPropagator(pupil, params)

# display pupil
# Note: It requires a rotational invariant pupil function
fig1, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.imshow(np.imag(pupil.return_pupil().detach().numpy()))
plt.show()
# display psf
fig2, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.imshow(propagator.compute_focus_field()[64, :, :].detach().numpy())
plt.show()
