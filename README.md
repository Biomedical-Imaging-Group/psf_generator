# PSF-Generator
***
Welcome to the psf-generator library!

This library implements various physical models that compute the point spread function (PSF) for microscopes. 
PSF characterizes the response of an imaging system to a point source of light and is crucial for tasks such as 
deconvolution, correction of aberrations, and characterization of the system.

We classify these models based on their physical property (scalar or vectorial) and numerical property (computed on a 
Cartesian or spherical coordinate system) and implement them as the following four
_propagators_

| Name of propagator             |         Other names         |
|--------------------------------|:---------------------------:|
| `ScalarCartesianPropagator`    | simple/scalar Fourier model |
| `ScalarSphericalPropagator`    |       Kirchhoff model       |
| `VectorialCartesianPropagator` |   vectorial Fourier model   |
| `VectorialSphericalPropagator` |     Richards-Wolf model     |

All of them can be derived from the Richards-Wolf integral under certain parameterization and conditions.
For details on the theory, please kindly refer to our paper
[Revisiting PSF models: unifying framework and high-performance implementation](todo:addlink) or the documentation: TO ADD LINK.

# Installation

## Basic Installation

```
pip install psf-generator
```

That's it for the basic intallation; you're ready to go!

## Developer Installation

If you're interested in experimenting with the code base, please clone the repository and install it using the following commands:
```
git clone git@github.com:Biomedical-Imaging-Group/psf_generator.git
cd psf_generator
pip install -e .
```

# Demos

Jupyter Notebook demos can be found under `demos/`.

# Napari Plugin
You can find our Napari plugin [here](https://github.com/Biomedical-Imaging-Group/napari-psfgenerator).

# Documentation
Documentation can be found here: TO ADD LINK

# Cite Us

TODO
