from .cartesian_propagator import CartesianPropagator
from .scalar_propagator import ScalarPropagator


class ScalarCartesianPropagator(ScalarPropagator, CartesianPropagator):

    def _get_input_field(self):
        return self.pupil.field


