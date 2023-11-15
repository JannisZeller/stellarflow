import numpy as np
import tensorflow as tf

from astropy.units import day as EarthDay
from astropy.time import Time as AstroTime
from astropy.coordinates import get_body_barycentric_posvel

from abc import ABC, abstractmethod

# from ..data import get_body_state
from ..data.states import get_astrotime_now

class Target(ABC):

    is_mooving: bool=False
    state_history: tf.Tensor
    state_vector: tf.Tensor

    def __init__(self, name: str):
        self.name = name

    @property
    def state_vector_numpy(self) -> np.ndarray:
        return self.state_vector.numpy()
    
    @property
    def position(self) -> tf.Tensor:
        return self.state_vector[:3]

    @property
    def position_numpy(self) -> tf.Tensor:
        return self.position.numpy()

    @property
    def state_history_numpy(self) -> tf.Tensor:
        return self.state_history.numpy()


    @abstractmethod
    def propagate(self):
        self.state_history = tf.concat(
            [self.state_history, [self.state_vector]], axis=0
        )


    @abstractmethod
    def reset(self):
        self.state_vector = self.state_history[0, ...]
        self.state_history = tf.expand_dims(self.state_vector, axis=0)




class FixedTarget(Target):

    def __init__(self, location: np.ndarray, name: str="target"):
        super().__init__(name)

        velocity = tf.zeros_like(location, dtype=tf.float32)
        self.state_vector = tf.concat(
            [tf.constant(location, dtype=tf.float32), velocity], axis=0
        )

        self.state_history = tf.expand_dims(self.state_vector, axis=0)


    def propagate(self):
        super().propagate()


    def reset(self):
        super().reset()



class OrbitTarget(Target):

    def __init__(
            self,
            planet: str,
            initial_time: AstroTime=None,
            tilt_angle: float=0,
            step_size: float=1.0,
            name: str="target"
        ):
        super().__init__(name)
        self.is_mooving = True

        self.planet = planet
        self.initial_time = initial_time
        self.step_size = step_size

        self.construct_rotation_matrix(tilt_angle)

        if initial_time is not None:
            self.set_state(initial_time)
        else:
            initial_time = get_astrotime_now()
            self.set_state(initial_time)

        self.state_history = tf.expand_dims(self.state_vector, axis=0)


    def construct_rotation_matrix(self, tilt_angle):
        theta_rad = (tilt_angle + 23.45) * np.pi / 180.
        rotation = np.array(
            [[1,             0,              0],
            [0,  np.cos(theta_rad), np.sin(theta_rad)],
            [0, -np.sin(theta_rad), np.cos(theta_rad)]]
        )
        stacked_rotation = np.vstack([
            np.hstack([rotation, np.eye(3, 3)]),
            np.hstack([np.eye(3, 3), rotation])
        ])
        self.rot_matrix = tf.constant(stacked_rotation, dtype=tf.float32)


    def apply_rotation(self):
        expanded_state = tf.expand_dims(self.state_vector, axis=0)
        expanded_state = expanded_state @ tf.transpose(self.rot_matrix)
        self.state_vector = tf.squeeze(expanded_state)


    def set_state(self, time: AstroTime):
        position = np.array(get_body_barycentric_posvel(self.planet, time)[0].xyz)
        velocity = np.array(get_body_barycentric_posvel(self.planet, time)[1].xyz)

        position = tf.constant(position, dtype=tf.float32)
        velocity = tf.constant(velocity, dtype=tf.float32)

        self.state_vector = tf.concat([position, velocity], axis=0)
        self.time = time

        self.apply_rotation()


    def propagate(self):
        next_time = self.time + self.step_size * EarthDay
        self.set_state(next_time)

        super().propagate()


    def reset(self):
        self.time = self.initial_time
        super().reset()
