"""
Benchmark runtime of the two scalar propagators on CPU and GPU for a range of number of pixels of the pupil.

The number of pixels of the PSF is fixed and only the one slice (focal plane) is computed.

"""
import os
import sys

from utils.handle_data import save_stats

module_path = os.path.abspath(os.path.join('')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)
from time import time

import numpy as np
import torch
from tqdm import tqdm

from propagators import ScalarCartesianPropagator, ScalarSphericalPropagator

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise Exception('Cuda Backend not found')

    # psf parameters
    n_pix_psf = 201
    na = 0.9
    wavelength = 632
    fov = 3000
    defocus = 0
    n_defocus = 1
    # define propagators
    propagators = [
        ScalarCartesianPropagator,
        ScalarSphericalPropagator
    ]
    # test parameters
    number_of_pupil_sizes = 9
    list_of_pupil_pixels = [int(item) for item in np.logspace(5, 13, number_of_pupil_sizes, base=2)]
    number_of_repetitions = 10
    devices = ["cpu", "cuda:0"]
    # file path to save statistics
    path = os.path.join('Results', 'data')

    for  n_pix_pupil in tqdm(list_of_pupil_pixels):
        for propagator in propagators:
            for device in devices:
                if 'cuda' in device:
                    torch.cuda.synchronize()
                runtime_list =[]
                average_runtime_list = []
                for _ in range(number_of_repetitions):
                    start_time = time()
                    propagator(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix_psf, wavelength=wavelength, na=na, fov=fov,
                               defocus_min=0, defocus_max=defocus, n_defocus=n_defocus, device=device).compute_focus_field()
                    runtime = time() - start_time
                    runtime_list.append(runtime)
                average_runtime_list.append((n_pix_pupil, sum(runtime_list) / number_of_repetitions))
                filename = f'{propagator}' + '_' + f'{device}'
                save_stats(average_runtime_list, filename, path)
