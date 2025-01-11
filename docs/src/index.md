# Usage
This library implements various physical models that compute the point spread function (PSF) for microscopes.

We classify these models based on their physical property (scalar or vectorial) and numerical property (computed on a 
Cartesian or spherical coordinate system) and implement them as the following four
_propagators_

| Name of propagator             |         Other names         |
|--------------------------------|:---------------------------:|
| `ScalarCartesianPropagator`    | simple/scalar Fourier model |
| `ScalarSphericalPropagator`    |       Kirchhoff model       |
| `VectorialCartesianPropagator` |   vectorial Fourier model   |
| `VectorialSphericalPropagator` |     Richards-Wolf model     |


We showcase how to use this library with `VectorialCartesianPropagator`.

## Define and run a propagator
To call a propagator, first import it as follows
```python
from psf_generator.propagators import VectorialCartesianPropagator
```

Then, give input parameters to the propagator.
All input parameters have a default value which can be overwritten by the user.
For example, to specify a numerical aperture of 1.2, simply do
```python
my_propagator = VectorialCartesianPropagator(na=1.2)
```

To specify many parameters, it is convenient to collect them in a dictionary
```python
kwargs = {
        'n_pix_pupil': 201,
        'n_pix_psf': 256,
        'wavelength': 600,
        'na': 1.4,
        'fov': 2000,
        'defocus_min': -4000,
        'defocus_max': 4000,
        'n_defocus': 200,
        'e0x': 1.0,
        'e0y': 1j,
        'gibson_lanni': True
    }
my_propagator = VectorialCartesianPropagator(**kwargs)
```
For a detailed explanation on all the input parameters, refer to the [documentation](TODO:addlink).

Then, to compute the pupil, simply do

```python
pupil = my_propagator.get_input_field()
```

and the PSF
```python
psf = my_propagator.compute_focus_field()
```

Both electric fields `pupil` and `PSF` are a `torch.tesnor` of data type `complex64` of size (3, n_pix_pupil, n_pix_pupil) and 
(n_defocus, 3, n_pix_psf, n_pix_psf), respectively.

## Visualize the results
For a convenient visual check, we provide two functions
- `plot_pupil`: modulus and phase of the pupil of all the components of the electric field (3 for vectorial Cartesian and 1 for scalar Cartesian)
- `plot_psf`: modulus, phase and intensity of the PSF at three orthogonal planes (xy, yz, and xz)

Here is an example

```python
from psf_generator.utils.plots import plot_pupil, plot_psf

name = my_propagator.get_name()
plot_pupil(pupil=pupil, name_of_propagator=name, filepath=None)
plot_psf(psf=psf, name_of_propagator=name, quantity='modulus', filepath=None)
```

For PSF, you need to specify which quantity to plot by passing the keyword 'modulus' or 'phase' or 'intensity' to the
argument `quantity`.
By default, the three orthogonal planes are the central slice in each dimension.

If you would like to save the plot as a `.png` file, simply specify a proper `filepath`.

**Note**: `plot_pupil` only supports Cartesian propagators.

## Save data
To save or load the original data of pupil or PSF along with the input parameters to a desired destination (`filepath`)
for further analysis:

```python
from psf_generator.utils.handle_data import save_as_npy, save_image, load_from_npy, load_image

# save the parameters as a json file
my_propagator.save_parameters(json_filepath)

data = pupil
# save as .npy
save_as_npy(filepath, data)
# save as .tif
save_image(filepath, data)
```

Note that `save_image` will move the _channel_ dimension to the last one, e.g. the saved image of pupil will have shape
(n_pix_pupil, n_pix_pupil, 3) and psf (n_defocus, n_pix_psf, n_pix_psf, 3).

The saved data can be conveniently loaded via
```python
# load a .npy file
load_from_npy(filepath)
# load a .tif file
load_image(filepath)
```

## Complete demo
Here is a simple demo to compute the pupil and PSF and visualize the results.
Check 'demos/' for more examples.

```python
from psf_generator.propagators import VectorialCartesianPropagator
from psf_generator.utils.plots import plot_pupil, plot_psf

if __name__ == "__main__":
    kwargs = {
        'n_pix_pupil': 201,
        'n_pix_psf': 256,
        'wavelength': 600,
        'fov': 1000,
        'defocus_min': -50,
        'defocus_max': 50,
        'n_defocus': 50,
        'gibson_lanni': True
    }
    my_propagator = VectorialCartesianPropagator(**kwargs)

    # compute pupil
    pupil = my_propagator.get_input_field()

    # compute PSF
    psf = my_propagator.compute_focus_field()

    # visualize the modulus and phase of the pupil
    plot_pupil(pupil=pupil, name_of_propagator=my_propagator.get_name())

    # visualize the modulus, phase and intensity of the PSF
    for quantity in ['modulus', 'phase', 'intensity']:
        plot_psf(psf=psf, name_of_propagator=my_propagator.get_name(), quantity=quantity)
```