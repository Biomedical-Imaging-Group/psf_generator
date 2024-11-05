import math
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from psf_generator.utils.handle_data import save_stats_as_csv
from torch.special import bessel_j1

from psf_generator.utils.misc import convert_tensor_to_array

module_path = os.path.abspath(os.path.join('')) + '/src/'
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from src.psf_generator import ScalarCartesianPropagator, ScalarSphericalPropagator

def benchmark_scalar_accuracy_on_airy_disk(
        n_pix_psf: int = 201,
        wavelength: float = 632,
        na: float = 0.9,
        fov: int = 3000,
        refractive_index: float = 1.5,
        debug: bool = False
):
    """
    Benchmark the accuracy of the two scalar propagators compared to an Airy disk against the size of the pupil.
    The accuracy is measured by the mean squared error.
    """
    # define parameters
    kwargs = {
        'n_pix_psf': n_pix_psf,
        'wavelength': wavelength,
        'na': na,
        'fov': fov,
        'refractive_index': refractive_index
    }

    # define ground truth: Airy disk
    airy_disk_function = lambda x: torch.where(x > 1e-6, 2 * bessel_j1(x) / x, 1 - x ** 2 / 8)
    x = torch.linspace(- fov / 2, fov / 2, n_pix_psf)
    xx, yy = torch.meshgrid(x, x, indexing='ij')
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    k = 2 * math.pi / wavelength
    airy_disk_analytic = convert_tensor_to_array(airy_disk_function(k * rr * na / refractive_index))

    # define propagators
    propagator_types = [
        ScalarCartesianPropagator,
        ScalarSphericalPropagator
    ]
    # test parameters
    list_of_pixels = [int(math.pow(2, exponent) + 1) for exponent in range(5, 13)]
    number_of_repetitions = 10
    # file path to save statistics
    path = os.path.join('results', 'data', 'benchmark_accuracy')


    for propagator_type in propagator_types:
        average_accuracy_list = []
        for n_pix in list_of_pixels:
            print(propagator_type.__name__, n_pix)
            accuracy_list = []
            for _ in range(number_of_repetitions):
                if 'cartesian' in propagator_type.get_name():
                    propagator = propagator_type(n_pix_pupil=n_pix, sz_correction=False, **kwargs)
                elif 'spherical' in propagator_type.get_name():
                    propagator = propagator_type(n_pix_pupil=n_pix, cos_factor=True, **kwargs)
                else:
                    raise ValueError('incorrect propagator name')
                psf = convert_tensor_to_array(propagator.compute_focus_field())
                psf /= np.max(np.abs(psf))
                accuracy = np.sqrt(np.sum(np.abs(psf - airy_disk_analytic) ** 2))
                if debug:
                    fig, axes = plt.subplots(1, 3)
                    norm = plt.Normalize(0.0, 1.0)
                    for ax, image in zip(axes, [psf.squeeze(), airy_disk_analytic, psf.squeeze() - airy_disk_analytic]):
                        ax.imshow(np.abs(image), norm=norm, cmap='inferno')
                    plt.show()
                    print(accuracy)
                accuracy_list.append(accuracy)
            average_accuracy_list.append((n_pix, sum(accuracy_list) / number_of_repetitions))
        # save stats
        filename = f'{propagator_type.get_name()}'
        filepath = os.path.join(path, filename + '.csv')
        save_stats_as_csv(filepath, average_accuracy_list)


if __name__ == "__main__":
    benchmark_scalar_accuracy_on_airy_disk()
