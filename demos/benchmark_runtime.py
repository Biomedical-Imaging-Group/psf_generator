"""
Benchmark runtime of the two scalar propagators on CPU and GPU for a range of number of pixels of the pupil.

The number of pixels of the PSF is fixed and only the one slice (focal plane) is computed.

"""
import os
import sys

from utils.handle_data import save_stats_as_csv

module_path = os.path.abspath(os.path.join('')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)
from time import time

import numpy as np
import torch
from tqdm import tqdm

from propagators import ScalarCartesianPropagator, ScalarSphericalPropagator, VectorialCartesianPropagator, \
    VectorialSphericalPropagator

if __name__ == "__main__":
    # psf parameters
    kwargs = {
        'n_pix_psf': 201,
        'wavelength': 632,
        'na': 0.9,
        'fov': 3000,
        'defocus_min': 0,
        'defocus_max': 0,
        'n_defocus': 1,
    }
    # define propagators
    propagators = [
        ScalarCartesianPropagator,
        ScalarSphericalPropagator,
        VectorialCartesianPropagator,
        VectorialSphericalPropagator
    ]
    # test parameters
    number_of_pupil_sizes = 9
    list_of_pupil_pixels = [int(item) for item in np.logspace(5, 13, number_of_pupil_sizes, base=2)]
    number_of_repetitions = 10
    devices = ["cpu", "cuda:0"]
    # file path to save statistics
    path = os.path.join('results', 'data')

    for device in devices:
        if 'cuda' in device:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            else:
                continue
        for propagator in propagators:
            average_runtime_list = []
            for n_pix_pupil in tqdm(list_of_pupil_pixels):
                runtime_list =[]
                for _ in range(number_of_repetitions):
                    start_time = time()
                    propagator(n_pix_pupil=n_pix_pupil, device=device, **kwargs).compute_focus_field()
                    runtime = time() - start_time
                    runtime_list.append(runtime)
                average_runtime_list.append((n_pix_pupil, sum(runtime_list) / number_of_repetitions))
            # save stats
            filename = f'{propagator.get_name()}_{device}'
            filepath = os.path.join(path, filename + 'csv')
            save_stats_as_csv(average_runtime_list, filepath)
