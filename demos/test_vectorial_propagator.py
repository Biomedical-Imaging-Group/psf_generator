import os
import sys

import matplotlib.pyplot as plt
import numpy as np

module_path = os.path.abspath(os.path.join('..')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from params import Params
from pupil import VectorialPupil
from propagator import SimpleVectorial, ComplexVectorial

FIGSIZE = (8, 8)

user_input = {
    'wavelength': 600,
    'NA': 1.1,
    'refractive_index': None
}

params = Params(user_input)
pupil = VectorialPupil(params)
propagator_simple = SimpleVectorial(pupil, params)
propagator_complex = ComplexVectorial(pupil, params)

# display pupil
fig1, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.imshow(propagator_simple.compute_focus_field()[0, :, :].detach().numpy())
plt.show()
# display psf
fig2, ax = plt.subplots(1, 1, figsize=FIGSIZE)
ax.imshow(propagator_complex.compute_focus_field()[0, :, :].detach().numpy())
plt.show()
