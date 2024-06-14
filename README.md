# PSF Generator

This library implements various models for computing the 3D point spread function (PSF) in microscopy applications. The PSF characterizes the response of an imaging system to a point source of light and is crucial for tasks such as image deconvolution, aberration correction, and system characterization.

## Features

- Scalar and Vectorial Models: Includes both scalar and vectorial models for PSF computation, accounting for the vector nature of the electric field.
- Cartesian and Polar Parametrizations: Supports both Cartesian and polar parametrizations for efficient computation and integration with Fast Fourier Transforms (FFTs) and Bessel functions.
- Defocus Modeling (not complete yet): Enables modeling the PSF at different focal positions, accounting for defocus effects.
- Aberration Modeling (not complete yet): Incorporates aberrations due to refractive index mismatches and allows for custom aberrations parametrized by Zernike polynomials.
- Apodization and Envelope Factors: Supports apodization factors (e.g., Gaussian envelope) and energy conservation factors (e.g., $\sqrt{\cos\theta}$).
- Fresnel Transmission Coefficients (not yet): Includes vectorial models with Fresnel transmission coefficients for accurate modeling of polarization effects.

## To-do next

- Vectorial model (VS)
- Investigation of different ways to compute the integrals for the scalar polar propagator (JC)
- Start the benchmarking codes (JD)
- Generate a realistic defocused beam, double-check physical constants, send this data to Eric Bezzam for him to test his models (JD)
