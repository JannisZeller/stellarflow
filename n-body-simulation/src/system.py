from typing import Callable, Tuple
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.solver import equations_of_motion_solver_factory


class NBodySystem:
    """
    This class embodies a system of stellar bodies, which can be propagated
    via the step or simulate methods. The unit system is set to length in
    astronomical units, mass in sun-masses and time in earth days.
    The suitable gravitational constant is provided within the class.
    """

    astronomical_unit: float = 149597870700.0
    earth_day: float = 86400.0
    gravitational_constant: float = 0.0002959211565456235

    def __init__(
        self,
        initial_locations: np.ndarray,
        initial_velocities: np.ndarray,
        masses: np.ndarray,
    ):
        if initial_locations.shape != initial_velocities.shape:
            message = (
                "Locations and velocities must be of the same shape, but got"
                + f"locations.shape={initial_locations.shape} and"
                + f"velocities.shape={initial_velocities.shape}."
            )
            raise ValueError(message)

        if masses.shape[0] != initial_locations.shape[0] or masses.ndim != 1:
            message = (
                "Locations and masses must be of the same length and masses must"
                + f"be of shape (N,), but got locations.shape={initial_locations.shape}"
                + f"and masses.shape={masses.shape}."
            )
            raise ValueError(message)

        self.body_count = initial_locations.shape[0]

        self._mask_reps, self._mask_len, self._mask = self._create_mask(
            self.body_count
        )
        self.state_matrix = tf.Variable(
            np.concatenate([initial_locations, initial_velocities], axis=1),
            dtype=tf.float32,
        )
        self.state_history = tf.cast(
            tf.expand_dims(self.state_matrix, axis=0), dtype=tf.float32
        )
        self.masses = self._reshape_masses(masses)

    def _create_mask(self, N: int) -> Tuple[int, int, tf.Tensor]:
        """Reshaping Mask for vectorization. TODO: Creates a Mask of shape"""
        mask_reps = int(N)
        mask_len = int(mask_reps * mask_reps)
        mask = tf.tile([False] + (mask_reps) * [True], [mask_reps])[:mask_len]
        return mask_reps, mask_len, mask

    def _reshape_masses(self, masses: np.ndarray) -> tf.Tensor:
        """Reshaping Masses for vectorization."""
        if self._mask is None:
            raise ReferenceError("Mask was not created.")
        M = tf.Variable(masses, dtype=tf.float32)
        M = tf.tile(M, [self._mask_reps])
        M = tf.reshape(M, (-1, 1))
        M = tf.boolean_mask(M, self._mask, axis=0)
        return tf.reshape(M, (self._mask_reps, -1, 1))

    @tf.function
    def _pairwise_distances(self, X: tf.Tensor) -> tf.Tensor:
        """Helper function to calculate the pairwise distances of the locations
        matrix."""
        ## TODO: Could be possibly further optimized. The current solution
        #  calculates all N^2 distances, but N^2/2 would be sufficient when
        #  switching the signs correctly.
        return tf.expand_dims(X, axis=0) - tf.expand_dims(X, axis=1)

    @tf.function
    def compute_acceleration(self, Q: tf.Tensor) -> tf.Tensor:
        """Vectorized acceleration calculation. Acts on the 6D-Tensor state
        tensor Q and returns dQ which acts as the "velocity" for Q
        (but the acceleration is the actual stuff that is computed).
        """
        D = self._pairwise_distances(Q[:, :3])

        D = tf.reshape(D, (-1, 3))
        D = tf.boolean_mask(D, self._mask, axis=0)
        D = tf.reshape(D, (self._mask_reps, (self._mask_reps - 1), -1))

        ## Calculating |xi-xj|^(-3)
        d_inv_cube = tf.pow(
            tf.reduce_sum(D * D, axis=-1, keepdims=True) + self.smooth, -3.0 / 2.0
        )
        ## Calculating pairwise Acceleration ("target mass" irrelevant)
        dV = self.masses * D * d_inv_cube
        ## Summation over each other body
        dV = self.gravitational_constant * tf.reduce_sum(dV, axis=1)
        dQ = tf.concat([Q[:, 3:], dV], axis=-1)

        return dQ

    def simulate(
        self,
        steps: int,
        step_size: float = 1.,
        smooth: float = 1e-20,
        algorithm: str = "rk4",
    ) -> None:
        """Performing a fixed number of steps."""
        self.smooth = smooth
        solver = equations_of_motion_solver_factory(algorithm, step_size)
        for _ in tqdm(range(steps)):
            self._step(solver)

    def _step(self, solver: Callable):
        """Performing one integration timestep. Integrates the equations of
        motion using the provided "solver" Callable.
        """
        self.state_matrix = solver(self.state_matrix, self.compute_acceleration)
        self.state_history = tf.concat(
            [self.state_history, [self.state_matrix]], axis=0
        )
