import os

from utils.handle_data import load_stats_from_csv
from utils.plots import plot_benchmark_results

folder = os.path.join('results', 'data')
files = ['scalar_cartesian', 'scalar_spherical', 'vectorial_cartesian', 'vectorial_spherical']
device_names = ['cpu', 'gpu']
title = 'benchmark runtime of pupil sizes'
results = []
labels = []
for file in files:
    for device_name in device_names:
        labels.append(f'{file}_{device_name}')
        results.append(load_stats_from_csv(os.path.join(folder, f'{file}_{device_name}.csv')))

plot_benchmark_results(results, labels, title)