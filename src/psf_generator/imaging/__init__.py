# Copyright Biomedical Imaging Group, EPFL 2025

"""
Dipole imagers: the image of a radiating dipole formed by the detection path of a microscope.
"""
from .cartesian_dipole_imager import CartesianDipoleImager
from .dipole_imager import FRESNEL_MODES, DipoleImager
from .spherical_dipole_imager import SphericalDipoleImager

#: Concrete imagers by name (as returned by ``get_name()``), used by ``DipoleImager.from_dict``.
IMAGERS = {
    imager.get_name(): imager
    for imager in (SphericalDipoleImager, CartesianDipoleImager)
}

__all__ = [
    'DipoleImager',
    'FRESNEL_MODES',
    'IMAGERS',
    'CartesianDipoleImager',
    'SphericalDipoleImager',
]
