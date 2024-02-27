import os
import sys

import matplotlib.pyplot as plt
import numpy as np

module_path = os.path.abspath(os.path.join('..')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from params import Params
from pupil import ScalarPupil
from propagator import FourierPropagator, SimpleVectorial

FIGSIZE = (8, 8)

user_input = {
    'wavelength': 600,
    'NA': 1.1,
    'refractive_index': None
}

params = Params(user_input)
pupil = ScalarPupil(params)
propagator = FourierPropagator(pupil, params)

# display pupil
fig1, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.imshow(np.imag(pupil.return_pupil().detach().numpy()))
plt.show()
# display psf
fig2, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.imshow(propagator.get_focus_field().detach().numpy())
plt.show()
