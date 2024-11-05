"""
Visualize the benchmark result of runtime and accuracy.

"""
import os

from src.psf_generator.propagators import *
from src.psf_generator.utils.handle_data import load_stats_from_csv
from src.psf_generator.utils.plots import plot_accuracy_benchmark_results, plot_runtime_benchmark_results


def compare_runtime(quantity: str):
    folder = os.path.join('results', 'data', 'benchmark_runtime')
    propagators = [
        ScalarCartesianPropagator,
        ScalarSphericalPropagator,
        VectorialCartesianPropagator,
        VectorialSphericalPropagator,
    ]
    device_names = ['cpu', 'gpu']
    title = f'Runtime benchmark against {quantity} sizes'
    filepath = os.path.join('results', 'plots', 'benchmark_runtime', f'benchmark_runtime_{quantity}_plot.png')
    results = []
    labels = []
    for propagator in propagators:
        for device_name in device_names:
            file = propagator.get_name()
            labels.append(f'{file}_{device_name}')
            results.append(load_stats_from_csv(os.path.join(folder, f'{file}_{device_name}_{quantity}.csv')))

    plot_runtime_benchmark_results(results=results, quantity=quantity, labels=labels, title=title, filepath=filepath)


def compare_accuracy():
    folder = os.path.join('results', 'data', 'benchmark_accuracy')
    propagators = [
        ScalarCartesianPropagator,
        ScalarSphericalPropagator
    ]
    title = 'Accuracy benchmark against pupil sizes'
    filepath = os.path.join('results', 'plots', 'benchmark_accuracy', 'benchmark_accuracy_plot.png')
    results = []
    labels = []
    for propagator in propagators:
        file = propagator.get_name()
        labels.append(f'{file}')
        results.append(load_stats_from_csv(os.path.join(folder, f'{file}.csv')))

    plot_accuracy_benchmark_results(results=results, labels=labels, title=title, filepath=filepath)

if __name__ == "__main__":
    for quantity in ['pupil', 'psf']:
        compare_runtime(quantity=quantity)
    compare_accuracy()
