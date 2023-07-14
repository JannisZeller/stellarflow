import numpy as np
import tensorflow as tf

from .sunsystem import SunSystem


class Walker:
    """
    Class for "test-bodies", e. g. spacecrafts or planets that are meant
    to be influenced by some reference-syste (instance of SunSystem`)
    """
    def __init__(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        reference_system: SunSystem=None,
        name: str="walker",
        mass: float=None
    ):
        position = tf.constant(initial_position, dtype=tf.float32)
        velocity = tf.constant(initial_velocity, dtype=tf.float32)
        self.state_vector = tf.concat([position, velocity], axis=-1)
        self.state_history = tf.expand_dims(self.state_vector, axis=0)
        self.mass = mass
        self.mass_history = [mass]
        self.name = name
        if reference_system is not None:
            self.reference_system = reference_system

    @property
    def state_vector_numpy(self) -> np.ndarray:
        return self.state_vector.numpy()


    @property
    def state_history_numpy(self) -> np.ndarray:
        return self.state_history.numpy()


    @property
    def position(self) -> np.ndarray:
        return self.state_vector[:3]


    @property
    def position_numpy(self) -> np.ndarray:
        return self.position.numpy()


    def _reset(self):
        """Resets the walker to the initial state.
        TODO: Later maybe randomly initialize the walker.
        """
        self.state_vector = self.state_history[0, ...]
        self.state_history = tf.expand_dims(self.state_vector, 0)


    def set_reference_system(self, reference_system: SunSystem):
        self.reference_system = reference_system


    @tf.function
    def compute_acceleration(self, state_vector):
        dx = state_vector[3:]
        dv = self.reference_system.gravitational_field(state_vector[:3])
        dq = tf.concat([dx, dv], axis=0)
        return dq


    def propagate(self, solver):
        self.state_vector = solver(self.state_vector, self.compute_acceleration)
        self.state_history = tf.concat(
            [self.state_history, [self.state_vector]], axis=0
        )
