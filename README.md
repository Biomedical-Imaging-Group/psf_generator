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

## Beyond the focus field: imaging and modalities

The propagators describe the illumination (focusing) path. Two more layers build on them:

- `psf_generator.imaging` computes the image of a radiating dipole through the detection path of the
  microscope (objective, coverslip and immersion medium, tube lens): `SphericalDipoleImager` (Bessel
  integrals, fast) and `CartesianDipoleImager` (chirp Z transform, any pupil aberration). The reverse path
  differs from focusing by its apodization, Jacobian and Fresnel factors, see the documentation.
- `psf_generator.modalities` combines illumination, sample and detection into the image recorded by a
  technique: `ISCATMicroscope`, `COBRIMicroscope` and `DarkFieldMicroscope` image a Rayleigh `Particle`
  interfering with a reference wave (interferometric scattering microscopy). Other modalities (confocal,
  image scanning microscopy, ...) can be added as subclasses of `Modality`.

```python
from psf_generator.modalities import ISCATMicroscope, Particle

gold = Particle(radius=15, permittivity=-3.7328 + 2.7725j)   # 30 nm gold at 517.5 nm
microscope = ISCATMicroscope(gold, wavelength=517.5, na=1.3, n_s=1.33, z_focus=1000, pix_size=40, n_pix_psf=101)
contrast = microscope.compute_contrast(positions=[(0, 0, z) for z in range(0, 2001, 50)])  # (41, 101, 101) iPSF
```

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
