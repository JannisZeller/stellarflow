import numpy as np
import tensorflow as tf
from astropy.units import day as EarthDay
from astropy.time import Time as AstroTime
from datetime import date

from .states import cartesians_to_array, get_cartesian_positions
from .data import get_masses


class StateError(Exception):
    pass


class SunSystem:
    """
    A base class for describing the sun system as a system of massive bodies.
    Provides a gravitational field for calculating accellearation of "test"-
    bodies at different locations.

    Properties
    ----------
    gravitational_constant: float = 0.0002959211565456235
        Gravitational constant for length in astronomical units, mass in
        sun-masses and time in earth days.
    smooth: float = 1e-20
        Smoothing the gravitational field to prevent 0-division.
    """

    gravitational_constant: float = 0.0002959211565456235
    smooth: float = 1e-20


    def __init__(
            self,
            bodies: list[str],
            initial_time: AstroTime=None,
            step_size: float=1.0,
            add_sun: bool=True
        ):
        """
        Constructs a `SunSystem` object for a `Walker` object to move in.

        Parameters
        ----------
        bodies: list[str]
            List of body names (like "earth", "sun" etc.) to be included in the
            system. Must be available `data.get_masses` (you can easily add
            more there yourself) and - more importantly - in Astropy's
            `get_body_barycentric_posvel`.
        initial_time: astropy.time.Time=None
            An initial time for the propagation of the system. If `None` the
            current date is used.
        step_size: float=1.0
            Step size for the propagation of the system in Earth-Days.
        add_sun: bool=True
            If `False` the sun is not added to the system, even if it is not
            included in `bodies`. Otherwise it will get added automatically.
        """
        if "sun" not in bodies and add_sun:
            bodies.append("sun")
        self.bodies = bodies
        self.n_bodies = len(bodies)
        self.masses = get_masses(bodies)
        self.masses_tensor = tf.reshape(
            tf.constant(self.masses, dtype=tf.float32),
            (-1, 1)
        )

        self.step_size = step_size
        if initial_time is not None:
            self.set_state(initial_time)
        else:
            now = date.today()
            initial_time = AstroTime(now.strftime(r'%Y-%m-%d %H:%M'), scale="utc")
            self.set_state(initial_time)
        self.positions_history = tf.expand_dims(self.positions_tensor, axis=0)
        self.initial_time = initial_time


    def __repr__(self):
        return f"<SunSystem - {self.bodies}>"


    def __str__(self):
        return f"SunSystem with bodes {self.bodies}"


    def _reset(self):
        """Resets the system to the initial state.
        TODO: Later maybe randomly initialize the initial time.
        """
        self.set_state(self.initial_time)
        self.positions_history = tf.expand_dims(self.positions_tensor, axis=0)


    def set_smooth(self, smooth: float):
        self.smooth = smooth


    def compute_positions(self, time: AstroTime, verbose: bool=False) -> np.ndarray:
        """Returns the positions of the internal bodies at a specific time.
        """
        cartesian_positions = get_cartesian_positions(self.bodies, time)
        positions_array = cartesians_to_array(cartesian_positions)
        if verbose:
            print(f"Positions at {time}:")
            print(positions_array)
        return positions_array


    def set_state(self, time: AstroTime):
        """Set the time and positions as the systems state.
        """
        positions = self.compute_positions(time, verbose=False)
        self.positions_tensor = tf.constant(positions, dtype=tf.float32)
        self.time = time


    def propagate(self):
        if not hasattr(self, 'time'):
            raise StateError("The system has no (initial) time set yet.")
        next_time = self.time + self.step_size * EarthDay
        self.set_state(next_time)
        self.positions_history = tf.concat(
            [self.positions_history, [self.positions_tensor]], axis=0
        )


    @tf.function
    def gravitational_field(self, x: tf.Tensor):
        d_vectors = self.positions_tensor - x
        distances = tf.pow(d_vectors, 2)
        # Calculating |xi-xj|^(-3)
        d_inv_cube = tf.pow(
            tf.reduce_sum(distances, axis=-1, keepdims=True) + self.smooth,
            -1.5
        )
        forces = (
            self.gravitational_constant *
            self.masses_tensor *
            d_vectors *
            d_inv_cube
        )
        force = tf.reduce_sum(forces, axis=0)
        return force
