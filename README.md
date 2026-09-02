# PSF-Generator

[![MIT License](https://img.shields.io/github/license/Biomedical-Imaging-Group/psf_generator)](https://github.com/Biomedical-Imaging-Group/psf_generator/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/psf-generator.svg?color=green)](https://pypi.org/project/psf-generator)
[![Python Version](https://img.shields.io/pypi/pyversions/psf-generator.svg?color=green)](https://python.org)
[![CI](https://github.com/Biomedical-Imaging-Group/psf_generator/actions/workflows/ci.yml/badge.svg)](https://github.com/Biomedical-Imaging-Group/psf_generator/actions/workflows/ci.yml)

***
Welcome to the psf-generator library!

This library contains a high-performance PyTorch implementation of precise physical models to compute the point spread function (PSF) of optical microscopes. 
The PSF characterizes the response of an imaging system to a point source and is crucial for tasks such as deconvolution, correction of aberrations, and characterization of the system.

We classify these models in two types—scalar or vectorial—and in both cases the PSF integral can be computed in Cartesian or spherical coordinate systems. 
This results in the following four _propagators_

| Name of propagator             |         Other names         |
|--------------------------------|:---------------------------:|
| `ScalarCartesianPropagator`    | simple/scalar Fourier model |
| `ScalarSphericalPropagator`    |       Kirchhoff model       |
| `VectorialCartesianPropagator` |   vectorial Fourier model   |
| `VectorialSphericalPropagator` |     Richards-Wolf model     |

For details on the theory, please refer to our paper
[here](https://doi.org/10.1111/jmi.70045).

# Documentation
Documentation can be found here: https://psf-generator.readthedocs.io/

# Installation

## Basic Installation

```
pip install psf-generator
```

That's it for the basic installation; you're ready to go!

## Developer Installation

If you're interested in experimenting with the code base, please clone the repository and install it using the following commands:
```
git clone git@github.com:Biomedical-Imaging-Group/psf_generator.git
cd psf_generator
pip install -e .
```

# Demos

Jupyter Notebook demos and Python scripts can be found under `demos/`.

# Development

Install the package with the test extras, then run the test suite and the linter:
```
pip install -e ".[test]"
pytest
ruff check --select F src/ tests/
```
The same checks run in continuous integration on every push and pull request.
Behaviour changes between releases are listed in `CHANGELOG.md`.

# Napari Plugin

You can find our Napari plugin [here](https://github.com/Biomedical-Imaging-Group/napari-psfgenerator).

# Cite Us

Liu, Y., Stergiopoulou, V., Chuah, J., Bezzam, E., Both, G.-J., Unser, M., Sage, D., & Dong, J. (2025).
Revisiting PSF models: Unifying framework and high-performance implementation. _Journal of Microscopy_, 1–13.
