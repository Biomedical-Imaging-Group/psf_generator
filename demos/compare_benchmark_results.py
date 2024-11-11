"""
Visualize the benchmark result of runtime and accuracy.

"""
import glob
import os

import matplotlib.pyplot as plt

from src.psf_generator.propagators import *
from src.psf_generator.utils.handle_data import load_stats_from_csv

_FIG_SIZE = 6
_TITLE_SIZE = 16
_LABEL_SIZE = 14
_TICK_SIZE = 12
lw = 1
markersize = 6


def plot_accuracy_benchmark_results(
        results: list,
        labels: list,
        title: str,
        filepath: str = None
):
    """
    Plot results of the accuracy benchmarking.

    Parameters
    ----------
    results : list
        List that contains tuples of the variable to be benchmarked and the resulting value.
    labels : str
        Label/name of the data.
    title : str
        Title of the figure.
    filepath : str, optional
        Path to save the figure. Default is None and figure is not saved.

    """
    figure, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE * 1.5, _FIG_SIZE))
    colors = ['red', 'blue']
    for result, label, color in zip(results, labels, colors):
        x, y = zip(*result)
        ax.loglog(x, y, label=label, ls='solid', marker='.', markersize=markersize, lw=lw, color=color)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.set_xlabel('Numerical size of the pupil / pixels', fontsize=_LABEL_SIZE)
    ax.set_ylabel('Error', fontsize=_LABEL_SIZE)
    xs, _ = zip(*results[0])
    xticks = [entry - 1 for entry in xs]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.legend(fontsize=_TICK_SIZE)

    plt.grid(color='gray', ls='dotted', lw=lw)
    figure.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()


def plot_runtime_benchmark_results(
        results: list,
        quantity: str,
        labels: list,
        title: str,
        filepath: str = None
):
    """
    Plot results of the runtime benchmarking.

    Parameters
    ----------
    results : list
        List that contains tuples of the variable to be benchmarked and the resulting value.
    quantity: str
        Quantity on the x-axis of the plot. Choose from 'pupil' or 'psf'.
    labels : str
        Label/name of the data.
    title : str
        Title of the figure.
    filepath : str, optional
        Path to save the figure. Default is None and figure is not saved.

    """
    figure, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE*1.5, _FIG_SIZE))
    colors = ['red', 'red', 'blue', 'blue', 'orange', 'orange', 'green', 'green']
    for result, label, color in zip(results, labels, colors):
        x, y = zip(*result)
        if 'cpu' in label:
            ls = 'solid'
        else:
            ls = 'dashed'
        ax.loglog(x[1:], y[1:], label=label, ls=ls, marker='.', markersize=markersize, lw=lw, color=color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_ylabel('Time / s', fontsize=_LABEL_SIZE)

    ax.legend(fontsize=_TICK_SIZE)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.set_xlabel(f'Numerical size of the {quantity} / pixels', fontsize=_LABEL_SIZE)

    plt.grid(color='gray', ls='dotted', lw=lw)
    figure.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()


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
    title = 'Accuracy benchmark against pupil sizes'
    filepath = os.path.join('results', 'plots', 'benchmark_accuracy', 'benchmark_accuracy_plot.png')
    results = []
    labels = []
    for file in sorted(glob.glob(os.path.join(folder, '*.csv'))):
        label = os.path.basename(file).removesuffix('.csv').removeprefix('scalar_')
        if label not in ('cartesian', 'spherical_simpsons_rule'):
            continue
        if label != 'cartesian':
            parts = label.split('_')
            label = f'{parts[0]}, {parts[1]} {parts[2]}'
        labels.append(label)
        results.append(load_stats_from_csv(file))

    plot_accuracy_benchmark_results(results=results, labels=labels, title=title, filepath=filepath)

if __name__ == "__main__":
    for quantity in ['pupil', 'psf']:
        compare_runtime(quantity=quantity)
    compare_accuracy()
