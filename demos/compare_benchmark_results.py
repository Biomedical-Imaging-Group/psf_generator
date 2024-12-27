"""
Visualize the benchmark result of runtime and accuracy.

"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from src.psf_generator.propagators import *
from src.psf_generator.utils.handle_data import load_stats_from_csv

_FIG_SIZE = 6
_TITLE_SIZE = 12
_LABEL_SIZE = 12
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
    figure, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE, _FIG_SIZE))
    colors = ['red', 'green', 'blue']
    xs = np.array([8, 16, 32, 64, 128, 256, 512, 1024]) + 1
    h2 = results[0][0][1] * (xs / xs[0]) ** (-2)
    h1 = results[1][0][1] * (xs / xs[0]) ** (-1)
    h4 = results[-1][0][1] * (xs / xs[0]) ** (-4)
    n_terms = len(xs)
    for result, label, color in zip(results, labels, colors):
        x, y = zip(*result)
        ax.loglog(x[:n_terms], y[:n_terms], label=label, ls='solid', marker='.', markersize=markersize, lw=lw, color=color)

    ax.loglog(xs[:n_terms], h1, label='$O(h)$', ls='dotted', lw=lw, color='green')
    ax.loglog(xs[:n_terms], h2, label='$O(h^2)$', ls='dotted', lw=lw, color='red')
    ax.loglog(xs[:n_terms], h4, label='$O(h^4)$', ls='dotted', lw=lw, color='blue')
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.set_xlabel('Pupil size', fontsize=_LABEL_SIZE)
    ax.set_ylabel('Error', fontsize=_LABEL_SIZE)
    xs, _ = zip(*results[0])
    xticks = [entry - 1 for entry in xs[:n_terms]]
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
        letter: str,
        show_legend: bool = True,
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
    letter: str
        Labeling for the figure, e.g. '(a)'.
    show_legend: bool, optional
        Whether to show the legend. Default is True.
    filepath : str, optional
        Path to save the figure. Default is None and figure is not saved.

    """
    figure, ax = plt.subplots(1, 1, figsize=(_FIG_SIZE, 0.6*_FIG_SIZE))
    colors = ['red', 'blue', 'red', 'blue']
    for result, label, color in zip(results, labels, colors):
        x, y = zip(*result)
        if 'scalar' in label:
            ls = 'dotted'
        else:
            ls = 'solid'
        ax.loglog(x[1:7], y[1:7], label=label.replace("_", " "), ls=ls, marker='.', markersize=markersize, lw=lw, color=color)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.set_yticks([1e-3, 1e-2, 1e-1])
        ax.set_xticks([32, 64, 128, 256, 512, 1024])
        ax.set_xticklabels([32, 64, 128, 256, 512, 1024])
        ax.set_ylabel('Time (s)', fontsize=_LABEL_SIZE)
    if show_legend:
        ax.legend(fontsize=_TICK_SIZE)
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.set_xlabel(f'{quantity.title()} size', fontsize=_LABEL_SIZE)
    ax.text(0.05, 0.05, letter, color='black', fontsize=_LABEL_SIZE, transform=ax.transAxes)

    plt.grid(color='gray', ls='dotted', lw=lw)
    figure.tight_layout()
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        figure.savefig(filepath)
    plt.show()


def compare_runtime(quantity: str, device_name: str, letter: str, show_legend: bool = True):
    folder = os.path.join('results', 'data', 'benchmark_runtime')
    propagators = [
        ScalarCartesianPropagator,
        ScalarSphericalPropagator,
        VectorialCartesianPropagator,
        VectorialSphericalPropagator,
    ]
    title = f'{device_name.upper()}'
    filepath = os.path.join('results', 'plots', 'benchmark_runtime', f'benchmark_runtime_{quantity}_{device_name}.pdf')
    results = []
    labels = []
    for propagator in propagators:
        file = propagator.get_name()
        labels.append(f'{file}')
        results.append(load_stats_from_csv(os.path.join(folder, f'{file}_{device_name}_{quantity}.csv')))

    plot_runtime_benchmark_results(results=results, quantity=quantity, labels=labels, title=title, letter=letter,
                                   show_legend=show_legend, filepath=filepath)


def compare_accuracy():
    folder = os.path.join('results', 'data', 'benchmark_accuracy')
    title = ''
    filepath = os.path.join('results', 'plots', 'benchmark_accuracy', 'benchmark_accuracy_plot.pdf')
    results = []
    labels = []
    for file in sorted(glob.glob(os.path.join(folder, '*.csv'))):
        label = os.path.basename(file).removesuffix('.csv').removeprefix('scalar_')
        if label not in ('cartesian', 'spherical_riemann_rule', 'spherical_simpsons_rule'):
            continue
        if label != 'cartesian':
            parts = label.split('_')
            label = f'{parts[0]}, {parts[1]} {parts[2]}'
        labels.append(label)
        results.append(load_stats_from_csv(file))

    plot_accuracy_benchmark_results(results=results, labels=labels, title=title, filepath=filepath)

if __name__ == "__main__":
    letters = ['(a)', '(b)']
    for quantity in ['pupil']:
        for device_name, letter in zip(['cpu', 'gpu'], letters):
            show_legend = True if device_name == 'cpu' else False
            compare_runtime(quantity=quantity, device_name=device_name, letter=letter, show_legend=show_legend)
    compare_accuracy()
