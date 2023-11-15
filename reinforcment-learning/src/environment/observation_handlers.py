from tf_agents.specs import array_spec

from abc import abstractmethod

from ._base import BaseHandler


class ObservationHandler(BaseHandler):

    @abstractmethod
    def construct_observation_spec(self):
        pass

    @abstractmethod
    def set_current_state(self, is_initial=False):
        if is_initial:
            self._initial_state = self.state




class StateAndDiffObservation(ObservationHandler):

    def construct_observation_spec(self):
        self._observation_spec = {
            "walker-state": array_spec.ArraySpec(
                self.walker.state_vector.shape,
                dtype=self.walker.state_vector_numpy.dtype,
                name="walker-state"
            ),
            "diff-to-target": array_spec.ArraySpec(
                (3,),
                dtype=self.target.position.numpy().dtype,
                name="target-state"
            )
        }


    def set_current_state(self, is_initial=False):
        target = self.target.position
        position = self.walker.position

        diff_to_target = target - position
        self.state = {
            "walker-state": self.walker.state_vector_numpy,
            "diff-to-target": diff_to_target.numpy()
        }
        super().set_current_state(is_initial)



class GravityObservation(ObservationHandler):

    def construct_observation_spec(self):
        self._observation_spec = {
            "gravity": array_spec.ArraySpec(
                self.walker.position.shape,
                dtype=self.walker.state_vector_numpy.dtype,
                name="gravity"
            ),
            "walker-state": array_spec.ArraySpec(
                self.walker.state_vector.shape,
                dtype=self.walker.state_vector_numpy.dtype,
                name="walker-state"
            ),
            "target": array_spec.ArraySpec(
                (3,),
                dtype=self.target.position_numpy.dtype,
                name="target"
            )
        }


    def set_current_state(self, is_initial=False):
        gravity = self.walker.compute_acceleration(
            self.walker.state_vector
        )
        self.state = {
            "gravity": gravity[3:].numpy(),
            "walker-state": self.walker.state_vector_numpy,
            "target": self.target.position_numpy
        }
        super().set_current_state(is_initial)


class WalkerTargetPositionsObservation(ObservationHandler):

    def construct_observation_spec(self):
        self._observation_spec = {
            "walker-state": array_spec.ArraySpec(
                self.walker.state_vector.shape,
                dtype=self.walker.state_vector_numpy.dtype,
                name="walker-state"
            ),
            "target-state": array_spec.ArraySpec(
                self.target.state_vector.shape,
                dtype=self.target.state_vector_numpy.dtype,
                name="target"
            )
        }


    def set_current_state(self, is_initial=False):
        self.state = {
            "walker-state": self.walker.state_vector_numpy,
            "target-state": self.target.state_vector_numpy
        }
        super().set_current_state(is_initial)


class AllPositionsObservation(ObservationHandler):

    def construct_observation_spec(self):
        self._observation_spec = {
            "system-positions": array_spec.ArraySpec(
                self.system._positions.shape,
                dtype=self.system._positions.numpy().dtype,
                name="system-positions"
            ),
            "walker-state": array_spec.ArraySpec(
                self.walker.state_vector.shape,
                dtype=self.walker.state_vector_numpy.dtype,
                name="walker-state"
            ),
            "target": array_spec.ArraySpec(
                self.target.shape,
                dtype=self.target.position_numpy.dtype,
                name="target"
            )
        }


    def set_current_state(self, is_initial=False):
        self.state = {
            "system-positions": self.system._positions.numpy(),
            "walker-state": self.walker.state_vector_numpy,
            "target": self.position_numpy
        }
        super().set_current_state(is_initial)
