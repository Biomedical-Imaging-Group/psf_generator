# Copyright Biomedical Imaging Group, EPFL 2025
__version__ = '0.2.0'

from .imaging import CartesianDipoleImager, SphericalDipoleImager
from .modalities import COBRIMicroscope, DarkFieldMicroscope, ISCATMicroscope, Particle
from .propagators import (
    ScalarCartesianPropagator,
    ScalarSphericalPropagator,
    VectorialCartesianPropagator,
    VectorialSphericalPropagator,
)

__all__ = [
    'ScalarCartesianPropagator',
    'ScalarSphericalPropagator',
    'VectorialCartesianPropagator',
    'VectorialSphericalPropagator',
    'SphericalDipoleImager',
    'CartesianDipoleImager',
    'Particle',
    'ISCATMicroscope',
    'COBRIMicroscope',
    'DarkFieldMicroscope',
]
