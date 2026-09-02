# Copyright Biomedical Imaging Group, EPFL 2025

"""
Modalities: complete image-formation models of microscopy techniques.
"""
from .modality import Modality, build_imager
from .particle import Particle
from .scattering import (
    SCHEMES,
    COBRIMicroscope,
    DarkFieldMicroscope,
    ISCATMicroscope,
    ScatteringMicroscope,
)

#: Concrete modalities by name (as returned by ``get_name()``), used by ``Modality.from_dict``.
MODALITIES = {
    modality.get_name(): modality
    for modality in (ScatteringMicroscope, ISCATMicroscope, COBRIMicroscope, DarkFieldMicroscope)
}

__all__ = [
    'Modality',
    'MODALITIES',
    'Particle',
    'SCHEMES',
    'ScatteringMicroscope',
    'ISCATMicroscope',
    'COBRIMicroscope',
    'DarkFieldMicroscope',
    'build_imager',
]
