"""
Benchmark runtime of the four propagators on CPU and GPU for a range of number of pixels of the pupil or PSF.

The other parameters are fixed and only the one slice (focal plane) is computed.

"""
import math
import os
import sys
from time import time

module_path = os.path.abspath(os.path.join('')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

import torch

from src.psf_generator.propagators import *
from src.psf_generator.utils.handle_data import save_stats_as_csv


def benchmark_runtime_on_size(
        quantity: str,
        n_pix_pupil: int = 201,
        n_pix_psf: int = 201,
        wavelength: float = 632,
        na: float = 0.9,
        fov: int = 3000,
):
    """Benchmark the runtime against the size of the pupil or PSF."""
    # propagator parameters
    kwargs = {
        'wavelength': wavelength,
        'na': na,
        'fov': fov,
    }
    # define propagators
    propagator_types = [
        ScalarCartesianPropagator,
        ScalarSphericalPropagator,
        VectorialCartesianPropagator,
        VectorialSphericalPropagator,
    ]
    # test parameters
    list_of_pixels = [int(math.pow(2, exponent)) for exponent in range(4, 13)]
    number_of_repetitions = 10
    devices = ["cpu", "cuda:0"]
    # file path to save statistics
    path = os.path.join('results', 'data', 'benchmark_runtime')

    for device in devices:
        if 'cuda' in device:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            else:
                continue
        for propagator_type in propagator_types:
            average_runtime_list = []
            for n_pix in list_of_pixels:
                print(device, propagator_type.__name__, n_pix)
                runtime_list = []
                for _ in range(number_of_repetitions):
                    start_time = time()
                    if quantity == 'pupil':
                        propagator = propagator_type(n_pix_pupil=n_pix, n_pix_psf=n_pix_psf, device=device, **kwargs)
                    elif quantity == 'psf':
                        propagator = propagator_type(n_pix_pupil=n_pix_pupil, n_pix_psf=n_pix, device=device, **kwargs)
                    else:
                        raise ValueError(f"Unknown quantity {quantity}, choose 'pupil' or 'psf'")
                    propagator.compute_focus_field()
                    runtime = time() - start_time
                    runtime_list.append(runtime)
                average_runtime_list.append((n_pix, sum(runtime_list) / number_of_repetitions))
            # save stats

            device_name = 'gpu' if 'cuda' in device else 'cpu'
            filename = f'{propagator_type.get_name()}_{device_name}_{quantity}'
            filepath = os.path.join(path, filename + '.csv')
            save_stats_as_csv(filepath, average_runtime_list)


if __name__ == "__main__":
    for quantity in ['pupil', 'psf']:
        benchmark_runtime_on_size(quantity=quantity)
