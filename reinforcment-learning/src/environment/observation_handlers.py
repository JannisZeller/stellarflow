from tf_agents.specs import array_spec

from abc import ABC, abstractmethod


class ObservationHandler(ABC):
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
                dtype=self.walker.state_vector.numpy().dtype,
                name="walker-state"
            ),
            "diff-to-target": array_spec.ArraySpec(
                self.target.shape,
                dtype=self.target.numpy().dtype,
                name="target"
            )
        }

    def set_current_state(self, is_initial=False):
        diff_to_target = self.target - self.walker.state_vector[:3]
        self.state = {
            "walker-state": self.walker.state_vector.numpy(),
            "diff-to-target": diff_to_target.numpy()
        }
        super().set_current_state(is_initial)




class GravityObservation(ObservationHandler):
    def construct_observation_spec(self):
        self._observation_spec = {
            "gravity": array_spec.ArraySpec(
                self.walker.state_vector[3:].shape,
                dtype=self.walker.state_vector.numpy().dtype,
                name="gravity"
            ),
            "walker-state": array_spec.ArraySpec(
                self.walker.state_vector.shape,
                dtype=self.walker.state_vector.numpy().dtype,
                name="walker-state"
            ),
            "target": array_spec.ArraySpec(
                self.target.shape,
                dtype=self.target.numpy().dtype,
                name="target"
            )
        }

    def set_current_state(self, is_initial=False):
        gravity = self.walker.compute_acceleration(
            self.walker.state_vector
        )
        self.state = {
            "gravity": gravity[3:].numpy(),
            "walker-state": self.walker.state_vector.numpy(),
            "target": self.target.numpy()
        }
        super().set_current_state(is_initial)


class AllPositionsObservation(ObservationHandler):
    def construct_observation_spec(self):
        self._observation_spec = {
            "system-positions": array_spec.ArraySpec(
                self.system.positions_tensor.shape,
                dtype=self.system.positions_tensor.numpy().dtype,
                name="system-positions"
            ),
            "walker-state": array_spec.ArraySpec(
                self.walker.state_vector.shape,
                dtype=self.walker.state_vector.numpy().dtype,
                name="walker-state"
            ),
            "target": array_spec.ArraySpec(
                self.target.shape,
                dtype=self.target.numpy().dtype,
                name="target"
            )
        }

    def set_current_state(self, is_initial=False):
        self.state = {
            "system-positions": self.system.positions_tensor.numpy(),
            "walker-state": self.walker.state_vector.numpy(),
            "target": self.target.numpy()
        }
        super().set_current_state(is_initial)
