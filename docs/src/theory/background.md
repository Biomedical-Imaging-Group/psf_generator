# Background
Electromagnetic waves in optical systems are fundamentally described by the Helmholtz wave equations yet solving these equations in full generality is computationally intractable.
Various approximations and integral formulations have thus been developed.
The cornerstone for precise PSF models in high-NA systems is the Richards-Wolf integral {cite:p}`richards1959electromagnetic`, which can be viewed as a vectorial extension of the Debye integral {cite:p}`debye1908lichtdruck`.
The conditions of validity for the Richards-Wolf integral have been thoroughly discussed by {cite:p}`wolf1981conditions`.
While alternative light propagation models exist, the Huygens-Fresnel approach has been shown to be equivalent, to some extent, for PSF calculations {cite:p}`egner1999equivalence`. 

## The line of work based on Bessel functions
This has been the basis of a line of work for PSF models based on Bessel functions, which we will later refer to as the spherical parametrization of the Richards-Wolf integral.
This approach includes simpler formulations like the Kirchhoff model and more sophisticated vectorial representations {cite:p}`aguet2009super, Novotny_Hecht_2012`.
These models have progressively been refined by incorporating various correction factors to account for additional physical processes: the Gibson-Lanni model for spherical aberrations due to refractive index mismatch {cite:p}`gibson1991experimental` (later generalized in {cite:p}`torok1995electromagnetic,torok1997electromagnetic`), apodization factors for energy conservation {cite:p}`richards1959electromagnetic`, and Fresnel transmission coefficients for accurate interface modeling {cite:p}`aguet2009super`. Models using this spherical parametrization have been implemented in various software libraries in Java {cite:p}`kirshner20133` and Python {cite:p}`caprile2022pyfocus`, which makes it widely accessible to researchers, albeit with some limitations in computational efficiency and integration with deep-learning frameworks.

## The line of work based on Fourier transforms
Another line of work on PSF modeling is based on Fourier transforms, both in scalar {cite:p}`goodman2005introduction` and vectorial {cite:p}`leutenegger2006fast` formulations.
These models are based on a Cartesian parametrization of the underlying Richards-Wolf integral and they represent a more general counterpart of the spherical parametrization.
Recently, these high-NA Fourier models have been implemented in MATLAB {cite:p}`miora2024calculating` and Tensorflow as part of a PSF fitting library {cite:p}`liu2024universal`.
Adequate sampling of the Fourier transform is crucial for obtaining high-resolution PSFs and avoiding aliasing.
A common trick based on the chirp Z transform is usually implemented to achieve it {cite:p}`leutenegger2006fast,miora2024calculating,Liu:25`.