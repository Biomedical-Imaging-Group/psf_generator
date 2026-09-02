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

- `Propagator.x` and `Propagator.z`: physical lateral and axial coordinates (nm) of the PSF grid.
- `zernike_basis`, `zernike_polynomial`, `osa_index_to_nl`, `nl_to_osa_index` in `psf_generator.utils.zernike`;
  `create_zernike_aberrations` accepts a precomputed `basis`.

### Fixed

- `create_zernike_aberrations` with a single coefficient returned an array of the wrong shape.

### Removed

- `CartesianPropagator.zoom_factor` (replaced by `k_start` / `k_end`), `zernike.index_to_nl` and
  `zernike.zernike_nl` (renamed to `osa_index_to_nl` and `zernike_polynomial`).

### Packaging

- The version is single-sourced from `psf_generator.__version__` and shown in the documentation.

## 0.1.0 (2025-06-17)

First release on PyPI of the models described in the paper (tag `v0.1.0`).
