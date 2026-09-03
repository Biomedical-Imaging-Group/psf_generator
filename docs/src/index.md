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
        'na': 1.2,
        'pix_size': 10,
        'defocus_step': 40,
        'n_defocus': 201,
        'e0x': 1.0,
        'e0y': 0.0,
        'gibson_lanni': True
    }
my_propagator = VectorialCartesianPropagator(**kwargs)
```
For a detailed explanation on all the input parameters, refer to the API Documentation.

The PSF is sampled on a grid that is centred on pixel `n_pix_psf // 2` along both lateral axes, with a pitch of
`pix_size` nanometers, and on z-slices located at `(i - n_defocus // 2) * defocus_step` nanometers, so that the
optical axis and the focal plane are always sampled exactly. The physical coordinates of the grid are available as
`my_propagator.x` (lateral, in nm) and `my_propagator.z` (axial, in nm).

Then, to compute the pupil, simply do

```python
pupil = my_propagator.get_pupil()
```

and the PSF
```python
psf = my_propagator.compute_focus_field()
```

Both electric fields `pupil` and `psf` are a `torch.Tensor` of data type `complex64` of size
(1, 3, n_pix_pupil, n_pix_pupil) and (n_defocus, 3, n_pix_psf, n_pix_psf), respectively (the second axis holds the
three components of the electric field; it has length 1 for the scalar propagators).

## Visualize the results
For a convenient visual check, we provide two functions
- `plot_pupil`: modulus and phase of the pupil of all the components of the electric field (3 for vectorial Cartesian and 1 for scalar Cartesian)
- `plot_psf`: modulus, phase and intensity of the PSF at three orthogonal planes ($xy$, $yz$, and $xz$)

Here is an example

```python
from psf_generator.utils.plots import plot_pupil, plot_psf

name = my_propagator.get_name()
plot_pupil(pupil=pupil, name_of_propagator=name, filepath=None)
plot_psf(psf=psf, name_of_propagator=name, quantity='modulus', filepath=None)
```

For PSF, you need to specify which quantity to plot by passing 'modulus', 'phase', 'intensity', 'amplitude' or
'stationary_phase' (the phase with the plane-wave factor removed; requires `propagator=my_propagator`) to the
argument `quantity`.
By default, the three orthogonal planes are the central slice in each dimension.

If you would like to save the plot as a `.png` file, simply specify a proper `filepath`.

**Note**: `plot_pupil` only supports Cartesian propagators.

## Save data
To save or load the original data of pupil or PSF along with the input parameters to a desired destination (`filepath`)
for further analysis:

```python
from psf_generator.utils.handle_data import save_as_npy, save_image, load_from_npy, load_image

# save the parameters as a json file ...
my_propagator.save_parameters(json_filepath)
# ... and rebuild an identical propagator later, with or without knowing its type
same_propagator = VectorialCartesianPropagator.load_parameters(json_filepath)
from psf_generator.propagators import Propagator
same_propagator = Propagator.load_parameters(json_filepath)
# the same round trip is available in memory with to_dict() / from_dict()

data = pupil
# save as .npy
save_as_npy(filepath, data)
# save as .tif
save_image(filepath, data)
```

Note that `save_image` writes the array as it is, without reordering its axes: the saved image of a vectorial pupil
keeps the shape (3, n_pix_pupil, n_pix_pupil) and that of a PSF the shape (n_defocus, 3, n_pix_psf, n_pix_psf).
`load_image` reads it back with the same shape, dtype and values.

The saved data can be conveniently loaded via
```python
# load a .npy file
load_from_npy(filepath)
# load a .tif file
load_image(filepath)
```

## Imaging a dipole and simulating a modality

The propagators describe the illumination path (a pupil function focused into the sample). The *dipole imagers*
of `psf_generator.imaging` describe the reverse, detection path: the image of a dipole radiating at any position
in the sample, through the coverslip and the immersion medium (see the theory section "Imaging a dipole"). They
take the same optical parameters as the propagators, plus the axial position `z_focus` of the focal plane, and
return the image field of shape (n_positions, 3, n_pix_psf, n_pix_psf) for a batch of dipole positions:

```python
from psf_generator.imaging import SphericalDipoleImager

imager = SphericalDipoleImager(wavelength=520, na=1.4, n_s=1.33, z_focus=500, pix_size=40, n_pix_psf=101)
field = imager.compute_image(dipole=(1.0, 0.0, 0.0), positions=[(0, 0, 0), (0, 0, 500), (200, -100, 1000)])
```

A *modality* (`psf_generator.modalities`) combines the illumination, the response of the sample and the
detection into the image recorded by a microscopy technique. The interferometric scattering family images a
Rayleigh scatterer interfering with a reference wave:

```python
from psf_generator.modalities import ISCATMicroscope, Particle

gold = Particle(radius=15, permittivity=-3.7328 + 2.7725j)          # 30 nm gold nanoparticle at 517.5 nm
microscope = ISCATMicroscope(gold, wavelength=517.5, na=1.3, n_s=1.33, n_g=1.5, n_i=1.5,
                             z_focus=1000, pix_size=40, n_pix_psf=101, n_pix_pupil=201)
positions = [(0, 0, z) for z in range(0, 2001, 50)]                  # (x_p, y_p, z_p) in nm, z_p above the coverslip
image = microscope.compute_image(positions)                           # (41, 101, 101), units of the incident intensity
contrast = microscope.compute_contrast(positions)                     # (I - I_ref) / I_ref, the iPSF
reference, scattered = microscope.compute_fields(positions)           # complex fields
microscope.save_parameters('iscat.json')                              # and Modality.load_parameters('iscat.json')
```

`COBRIMicroscope` (transmission) and `DarkFieldMicroscope` share the same interface; `attenuation` scales the
reference wave, `e0x`/`e0y` set the illumination polarization and `imager='cartesian'` switches to the Cartesian
imager, which accepts any Zernike aberration of the detection path. See the notebook
`demos/notebooks/iscat_microscopy.ipynb` for a guided tour and `demos/scripts/iscat_demo.py` for a script.

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
        'na': 1.2,
        'pix_size': 10,
        'defocus_step': 40,
        'n_defocus': 201,
        'e0x': 1.0,
        'e0y': 0.0,
        'gibson_lanni': True
    }
    my_propagator = VectorialCartesianPropagator(**kwargs)

    # compute pupil
    pupil = my_propagator.get_pupil()

    # compute PSF
    psf = my_propagator.compute_focus_field()

    # visualize the modulus and phase of the pupil
    plot_pupil(pupil=pupil, name_of_propagator=my_propagator.get_name())

    # visualize the modulus, phase and intensity of the PSF
    for quantity in ['modulus', 'phase', 'intensity']:
        plot_psf(psf=psf, name_of_propagator=my_propagator.get_name(), quantity=quantity)
```