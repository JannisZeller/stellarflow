from typing import Callable, Tuple
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm


class NBodySystem:
    """
    This class embodies a system of stellar bodies, which can be propagated
    via the step or simulate methods. The unit system is set to length in
    astronomical units, an time in earth days. The suitable constants are
    provided within the class
    """

    astronomical_unit: float = 149597870700.0
    earth_day: float = 86400.0
    gravitational_constant: float = 0.0002959211565456235

    def __init__(
        self,
        initial_locations: np.ndarray,
        initial_velocities: np.ndarray,
        masses: np.ndarray,
        dt: float = 1.0,  # Step size in earth days
        smooth: float= 1e-20
    ):
        self.locations = initial_locations
        self.velocities = initial_velocities
        self.masses = masses
        self.dt = dt

        ## Pre-Creating mask for pairwise calculations later on
        self._mask_reps, self._mask_len, self._mask = self._create_mask(
            initial_locations.shape[0]
        )

        ## Combining locations and velocities to internal state tf.Tensor
        assert (
            initial_locations.shape == initial_velocities.shape
        ), f"Locations and velocities must be of the same shape, but got \
            locations.shape={initial_locations.shape} and \
            velocities.shape={initial_velocities.shape}."

        ## Setting up Q-Tensor for internal processing
        self._Q = tf.Variable(
            np.concatenate([initial_locations, initial_velocities], axis=1), dtype=tf.float32
        )
        self._Q_hist = tf.cast(tf.expand_dims(self._Q, axis=0), dtype=tf.float32)

        ## Storing the masses to a tf.Tensor
        assert (masses.shape[0] == initial_locations.shape[0] and masses.ndim == 1), (
            f"Locations and masses must be of the same length and masses must" +
            f"be of shape (N,), but got locations.shape={initial_locations.shape}" +
            f"and masses.shape={masses.shape}.")
        self._M = self._reshape_masses(masses)

        self.smooth = smooth


    ## Reshaping Masses for vectorization
    def _reshape_masses(self, masses: np.ndarray):
        M = tf.Variable(masses, dtype=tf.float32)
        M = tf.tile(M, [self._mask_reps])
        M = tf.reshape(M, (-1, 1))
        M = tf.boolean_mask(M, self._mask, axis=0)
        return tf.reshape(M, (self._mask_reps, -1, 1))


    ## Reshaping Mask for vectorization -> Which body is affected by which
    #  other?
    def _create_mask(self, N: int):
        mask_reps = int(N)
        mask_len = int(mask_reps * mask_reps)
        mask = tf.tile([False] + (mask_reps) * [True], [mask_reps])[:mask_len]
        return mask_reps, mask_len, mask


    ## Fully vectorized implementation of pairwise distances between the bodys
    @tf.function
    def _pairwise_distances(self, X: tf.Tensor) -> tf.Tensor:
        """Helper function to calculate the pairwise distances of the locations matrix."""
        ## TODO: Could be possibly further optimized. The current solution
        #  calculates all N^2 distances, but N^2/2 would be sufficient when
        #  switching the signs correctly.
        return tf.expand_dims(X, axis=0) - tf.expand_dims(X, axis=1)


    ## Vectorized acceleration calculation. Acts on the 6D-Tensor Q and returns
    #  dQ which acts as the "velocity" for Q (but the acceleration is the
    #  actual stuff that is computed).
    @tf.function
    def _acceleration(self, Q: tf.Tensor) -> tf.Tensor:
        ## Calculating Pairwise Distances
        D = self._pairwise_distances(Q[:, :3])

        ## Masking and reshaping X and M to drop self-differences:
        D = tf.reshape(D, (-1, 3))
        D = tf.boolean_mask(D, self._mask, axis=0)
        D = tf.reshape(D, (self._mask_reps, (self._mask_reps - 1), -1))

        ## Calculating |xi-xj|^(-3)
        d_inv_cube = tf.pow(
            tf.reduce_sum(D * D, axis=-1, keepdims=True) + self.smooth, -3.0 / 2.0
        )

        ## Calculating pairwise Acceleration ("target mass" irrelevant)
        dV = self._M * D * d_inv_cube

        ## Summation over each other body
        dV = self.gravitational_constant * tf.reduce_sum(dV, axis=1)

        ## Combining to R6 acceleration
        #  The own masses are not needed because of the formula in "../README.md"
        dQ = tf.concat([Q[:, 3:], dV], axis=-1)

        return dQ


    ## Runge-Kutta-Fehlberg solver for integrating dQ = dQ(Q) where dQ(Q) is the
    #  6D acceleration (v, a) where a is induced by the pairwise gravitation
    #  of the bodies.
    @tf.function
    def _solver_rk4(self, Q: tf.Tensor, dQ: Callable):
        dt = self.dt
        k1 = dQ(Q)
        k2 = dQ(Q + dt * k1 / 2.0)
        k3 = dQ(Q + dt * k2 / 2.0)
        k4 = dQ(Q + dt * k3)
        return Q + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


    @tf.function
    def _solver_rkf(self, Q: tf.Tensor, dQ: Callable):
        dt = self.dt
        k1 = dQ(Q)
        k2 = dQ(Q + dt * k1 / 4.0)
        k3 = dQ(Q + dt * k1 * 3.0 / 32.0 + dt * k2 * 9.0 / 32.0)
        k4 = dQ(
            Q
            + dt * k1 * 1932.0 / 2197.0
            - dt * k2 * 7200.0 / 2197.0
            + dt * k3 * 7296.0 / 2197.0
        )
        k5 = dQ(
            Q
            + dt * k1 * 439.0 / 216.0
            - dt * k2 * 8.0
            + dt * k3 * 3680.0 / 513
            - dt * k4 * 845.0 / 4104.0
        )
        k6 = dQ(
            Q
            - dt * k1 * 8.0 / 27.0
            + dt * k2 * 2.0
            - dt * k3 * 3544.0 / 2565.0
            + dt * k4 * 1859.0 / 4104.0
            - dt * k5 * 11.0 / 40.0
        )
        Q = Q + dt * (
            16.0 / 135.0 * k1
            + 6656.0 / 12825.0 * k3
            + 28561.0 / 56430.0 * k4
            - 9.0 / 50.0 * k5
            + 2.0 / 55.0 * k6
        )
        return Q


    ## Performing / integrating one timestep.
    def step(self, algo: str = "rk4") -> None:
        if algo == "rk4":
            Q = self._solver_rk4(self._Q, self._acceleration)
        if algo == "rkf":
            Q = self._solver_rkf(self._Q, self._acceleration)
        self._Q = Q
        self._Q_hist = tf.concat([self._Q_hist, [Q]], axis=0)


    ## Performing a fixed number of steps.
    def simulation(self, steps: int, algo: str = "rk4") -> None:
        for _ in tqdm(range(steps)):
            self.step(algo)


    ## Resetting the system.
    def _reset(self):  # This version for tfa-environment
        self._Q = self._Q_hist[0]
        self._Q_hist = [self._Q]


    def reset(self):
        self._reset()


    ## 3D-Plotting
    def plot_history_3d(self, n_sample=25, figsize=(5, 5)) -> None:
        if self._Q_hist.shape[1] > n_sample:
            sample = np.random.choice(
                np.arange(self._Q_hist.shape[1]), size=n_sample, replace=False
            )
        else:
            sample = range(self._Q_hist.shape[1])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        for idx in sample:
            ax.plot(
                self._Q_hist[:, idx, 0],
                self._Q_hist[:, idx, 1],
                self._Q_hist[:, idx, 2],
            )


    ## 2D-Plotting
    def plot_history_2d(
        self,
        ZSIZE: bool = False,
        n_sample: int = 100,
        figsize: Tuple[int, int] = (5, 5),
    ) -> None:
        def plot_single_2d(n, ZSIZE=ZSIZE):
            if ZSIZE:
                plt.scatter(
                    self._Q_hist[:, n, 0],
                    self._Q_hist[:, n, 1],
                    0.75
                    * np.clip(self._Q_hist[:, n, 2].numpy(), a_min=1e-10, a_max=np.inf),
                )
            else:
                plt.scatter(self._Q_hist[:, n, 0], self._Q_hist[:, n, 1])

        plt.figure(figsize=figsize)
        if self._Q_hist.shape[1] > n_sample:
            sample = np.random.choice(
                np.arange(self._Q_hist.shape[1]), size=n_sample, replace=False
            )
        else:
            sample = range(self._Q_hist.shape[1])
        for idx in sample:
            plot_single_2d(idx)
        plt.show()
