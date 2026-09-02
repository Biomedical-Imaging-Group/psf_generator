# Changelog

## 0.2.0 (unreleased)

### Changed

Numerical results differ slightly from 0.1.0 because of the first two items.

- The PSF grid is now pixel-centred: samples are located at `x[i] = (i - n_pix_psf // 2) * pix_size`, so
  neighbouring pixels are exactly `pix_size` apart and the optical axis goes through pixel `n_pix_psf // 2`
  (for even sizes as well). Previously both endpoints of the field of view were sampled, which made the
  pitch `pix_size * n / (n - 1)`.
- The z-slices are now located at `z[i] = (i - n_defocus // 2) * defocus_step`, so consecutive slices are
  exactly `defocus_step` apart and slice `n_defocus // 2` is the focal plane. Previously the half-range was
  `(defocus_step * n_defocus) // 2`, which gave a step of `defocus_step * n / (n - 1)` and, for odd
  `n_defocus`, no slice at z = 0.
- Zernike polynomials are computed in PyTorch (`psf_generator.utils.zernike`); the `zernikepy` dependency is
  removed. The convention is unchanged: OSA/ANSI single index, unnormalized modes, coefficients in radians.
  The basis is cached per propagator, so `update_zernike_coefficients` only costs a weighted sum.
- The centring correction of the chirp Z transform is exact for asymmetric sampling ranges.

### Added

- `psf_generator.imaging`: the detection path of the microscope. `SphericalDipoleImager` and
  `CartesianDipoleImager` compute the image of a radiating dipole (a fluorophore, or the dipole induced in a
  nanoparticle) located anywhere in the sample, through the coverslip and the immersion medium, with the
  geometric factors of the reverse path (`1/sqrt(cos)` apodization, no `1/s_z` Jacobian, reciprocal Fresnel
  coefficients, radiation pattern in the sample medium, evanescent components beyond the critical angle) and
  Zernike aberrations of the detection path; heights and lateral positions are batched. In a homogeneous medium
  the image equals the apodized focus field of the vectorial propagators.
- `psf_generator.modalities`: complete image-formation models. `ISCATMicroscope`, `COBRIMicroscope` and
  `DarkFieldMicroscope` (interferometric scattering, coherent bright-field and dark-field microscopy) image a
  Rayleigh `Particle` interfering with the reflected or transmitted illumination, with the optical paths of the
  reference and scattered waves, an optional attenuation of the reference and any illumination polarization;
  `compute_image`, `compute_contrast` (the iPSF) and `compute_fields`. `Modality` is the base class for further
  techniques (confocal, image scanning microscopy).
- `psf_generator.utils.parameters.Parametrized`: the JSON round trip (`to_dict`, `from_dict`,
  `save_parameters`, `load_parameters`) shared by the propagators, the imagers (`IMAGERS` registry, key
  `'imager'`) and the modalities (`MODALITIES` registry, key `'modality'`). The format written by the
  propagators is unchanged.
- Theory page "Imaging a dipole: the detection path" in the documentation and the demo script
  `demos/scripts/iscat_demo.py`.
- The propagators validate their arguments and raise a `ValueError` with a clear message for an unknown
  `device`, `n_pix_pupil < 2`, `n_pix_psf < 1`, `n_defocus < 1`, a non-positive `wavelength`, `pix_size` or
  `na`, and `na > n_i0` (which used to produce NaNs or a meaningless field silently; `na == n_i0` is allowed).
- `Propagator.x` and `Propagator.z`: physical lateral and axial coordinates (nm) of the PSF grid.
- `zernike_basis`, `zernike_polynomial`, `osa_index_to_nl`, `nl_to_osa_index` in `psf_generator.utils.zernike`;
  `create_zernike_aberrations` accepts a precomputed `basis`.
- `Propagator.to_dict()`, `Propagator.from_dict()` and `Propagator.load_parameters()`: a propagator can be rebuilt
  from its saved parameters (`save_parameters` writes the same dictionary). Called on the base class, the type is
  taken from the new `'propagator'` key; `psf_generator.propagators.PROPAGATORS` maps names to classes. Files
  written by 0.1.0 are still accepted.

### Fixed

- The Cartesian propagators raised `ZeroDivisionError` for `n_pix_psf = 1`: the chirp Z transform divided by
  `n_pix_psf - 1` although a single sample needs no step. A one-pixel PSF now returns the value at the optical
  axis, i.e. the centre pixel of a larger odd grid.
- The argument validation rejected NumPy scalars (`np.int64` is not a subclass of `int`), so a size taken from
  an array or from `np.arange` raised a `ValueError`; any `numbers.Real` is now accepted.
- `custom_fft2` and `custom_ifft2` silently returned `None` for an unsupported `norm`; they now raise a
  `ValueError`. A non-square requested output warns instead of printing to stdout.
- `plots.apply_disk_mask` built its grid with `np.linspace(0, n, n)` instead of `np.arange(n)`, so the disk was
  stretched by up to one pixel and was not symmetric on odd images.
- Documentation: `t_g` / `t_g0` are the thickness of the cover slip, not of the sample; the formula for the
  internal `t_i` (in the `Propagator` docstring and in the theory pages) carried a leading defocus term that
  the code does not have, since the library applies the defocus through the propagation kernel; the matrix in
  the `VectorialSphericalPropagator` docstring was garbled; `simpsons_rule` works with any odd number of
  samples, not only with `2^K + 1`.

- `save_image` wrote TIFF files whose axes tifffile had to guess: a scalar PSF stack of shape
  `(n_defocus, 1, x, y)` was refused with "not enough samples for RGB", and a vectorial stack
  `(n_defocus, 3, x, y)` was stored as planar RGB and read back transposed as `(n_defocus, x, y, 3)`. TIFF
  files are now written and read with an explicit layout, so any array round trips through
  `save_image` / `load_image` with the same shape, dtype (`complex64` included) and values.
- `save_image`, `save_as_npy`, `save_stats_as_csv`, `plot_pupil` and `plot_psf` raised
  `FileNotFoundError: ''` when given a bare filename with no directory part.

- The spherical propagators crashed with "Expected all tensors to be on the same device" on any non-CPU device
  as soon as a correction factor was requested (`apod_factor`, `envelope`, `gibson_lanni` or `cos_factor`):
  the sines and cosines of the polar angle were computed on the CPU. They now run on CUDA and MPS.

- The spherical propagators evaluated the Zernike modes at the wrong pupil radius. They sample the pupil
  uniformly in the polar angle, so sample `i` sits at the normalized radius `sin(theta_i) / sin(theta_max)`,
  but the modes were evaluated on an equispaced radius `i / (n_pix_pupil - 1)`, i.e. at `theta_i / theta_max`.
  At NA 1.4 in oil (`n = 1.5`, `theta_max = 69 deg`) the mid-pupil sample sat at radius 0.5 instead of 0.607,
  so a given set of coefficients described a different wavefront than in the Cartesian propagators: with a
  1.5 rad defocus (OSA index 4) or primary spherical (index 12) coefficient the normalized in-focus PSFs of
  `ScalarCartesianPropagator` and `ScalarSphericalPropagator` differed by 0.037 and 0.031 (max abs difference,
  against 0.0004 without aberration); they now differ by 0.0012 and 0.0006. `zernike_basis` takes an optional
  `rho` argument with the radius of every sample of a spherical mesh (the spherical propagators pass their own,
  stored in the new attribute `SphericalPropagator.rho`); without it the radius is equispaced as before.

- `create_zernike_aberrations` with a single coefficient returned an array of the wrong shape.
- `VectorialCartesianPropagator` accepts `sz_correction` and `custom_field`, and `VectorialSphericalPropagator`
  accepts `custom_field`, like their scalar counterparts.

- `save_parameters` no longer writes the derived values `refractive_index` and `t_i`, writes complex numbers as
  `[real, imag]` pairs instead of strings, and also stores `n_i0`, `sz_correction`, `special_phase_mask`
  (Cartesian), `cos_factor` and `integrator` (spherical).

### Removed

- `CartesianPropagator.zoom_factor` (replaced by `k_start` / `k_end`), `zernike.index_to_nl` and
  `zernike.zernike_nl` (renamed to `osa_index_to_nl` and `zernike_polynomial`).

### Packaging

- The version is single-sourced from `psf_generator.__version__` and shown in the documentation.
- `tifffile` is declared as an explicit dependency: `handle_data` now uses it directly to read and write TIFF
  files (it was already installed as a dependency of `scikit-image`).

## 0.1.0 (2025-06-17)

First release on PyPI of the models described in the paper (tag `v0.1.0`).
