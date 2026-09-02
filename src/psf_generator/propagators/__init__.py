# Copyright Biomedical Imaging Group, EPFL 2025

from .propagator import Propagator
from .scalar_cartesian_propagator import ScalarCartesianPropagator
from .scalar_spherical_propagator import ScalarSphericalPropagator
from .vectorial_cartesian_propagator import VectorialCartesianPropagator
from .vectorial_spherical_propagator import VectorialSphericalPropagator

#: Concrete propagators by name (as returned by ``get_name()``), used by ``Propagator.from_dict``.
PROPAGATORS = {
    propagator.get_name(): propagator
    for propagator in (
        ScalarCartesianPropagator,
        ScalarSphericalPropagator,
        VectorialCartesianPropagator,
        VectorialSphericalPropagator,
    )
}

__all__ = [
    'Propagator',
    'PROPAGATORS',
    'ScalarCartesianPropagator',
    'ScalarSphericalPropagator',
    'VectorialCartesianPropagator',
    'VectorialSphericalPropagator',
]
