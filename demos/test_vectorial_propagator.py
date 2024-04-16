import os
import sys

import matplotlib.pyplot as plt
import numpy as np

module_path = os.path.abspath(os.path.join('..')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from params import Params
from pupil import VectorialPupil
from propagator import Vectorial

FIGSIZE = (8, 8)

user_input = {
    'wavelength': 600,
    'NA': 1.1,
    'refractive_index': None
}

params = Params(user_input)
pupil = VectorialPupil(params)
propagator = Vectorial(pupil, params)

# display pupil
fig1, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.plot(pupil.return_pupil().detach()[0, :])
plt.show()

# display psf
fig2, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.imshow(propagator.compute_focus_field()[64, 0, :, :].detach().numpy())
plt.show()
