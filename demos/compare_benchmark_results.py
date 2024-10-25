import os

from src.psf_generator.propagators import *
from src.psf_generator.utils.handle_data import load_stats_from_csv
from src.psf_generator.utils.plots import plot_benchmark_results

if __name__ == "__main__":
    folder = os.path.join('results', 'data')
    propagators = [
        ScalarCartesianPropagator,
        ScalarSphericalPropagator,
        VectorialCartesianPropagator,
        VectorialSphericalPropagator
    ]
    device_names = ['cpu', 'gpu']
    title = 'Runtime benchmark against pupil sizes'
    filepath = os.path.join('results', 'plots', 'benchmark_plot.png')
    results = []
    labels = []
    for propagator in propagators:
        for device_name in device_names:
            file = propagator.get_name()
            labels.append(f'{file}_{device_name}')
            results.append(load_stats_from_csv(os.path.join(folder, f'{file}_{device_name}.csv')))

    plot_benchmark_results(results, labels, title, filepath)