from .cartesian_propagator import CartesianPropagator


class ScalarCartesianPropagator(CartesianPropagator):

    def _get_input_field(self):
        return self.pupil.field


