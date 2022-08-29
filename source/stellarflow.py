# %% Imports
#-------------------------------------------------------------------------------

from typing import Callable

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

#-------------------------------------------------------------------------------

## Next step could be the implementation of different "scales" combining step sizes and unit systems such that different types (solar systems, larger structures etc.) can be simulated.


# %% Implementation
# Implementation of the simulation class
#-------------------------------------------------------------------------------

class System():
    """This class embodies a system of stellar bodies, which can be propagated via the step or simulate methods.
    """

    _AU: float=149597870700.
    _ED: float=86400.

    def __init__(self, 
        locations: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        dt: float=1., # Step size in earth days
        smooth: float=1e-20, # Smoothing of distances when dividing
        _AU: float=149597870700., # Astronomical Units
        _ED: float=86400., # Earth Day
        _G: float=6.67430e-11 / 149597870700.**3. * 86400.**2. * 1.98847e30 # Gravitational constant in AU, ED, Sun Mass units
    ):
        self.locations  = locations
        self.velocities = velocities
        self.masses = masses
        self.dt  = dt
        self.smooth = smooth
        self._AU = _AU
        self._ED = _ED
        self._G  = _G

        ## The Internal State and Mass Tensors get already constructed here:
        #-------------------------------------------------------------------

        ## Pre-Creating mask for pairwise calculations later on
        self._mask_reps = int(locations.shape[0])
        self._mask_len  = int(self._mask_reps*self._mask_reps)
        self._mask = tf.tile([False] + (self._mask_reps)*[True], [self._mask_reps])[:self._mask_len] 

        ## Combining locations and velocities to internal state tf.Tensor
        assert locations.shape == velocities.shape, f"Locations and velocities must be of the same shape, but got locations.shape={locations.shape} and velocities.shape={velocities.shape}."
        Q = np.concatenate([locations, velocities], axis=1)
        self._Q = tf.Variable(Q, dtype=tf.float32)
        self._Q_hist = [Q]

        ## Storing the masses to a tf.Tensor
        assert masses.shape[0] == locations.shape[0] and masses.ndim == 1, f"Locations and masses must be of the same length and masses must be of shape (N,), but got locations.shape={locations.shape} and masses.shape={masses.shape}."
        self._M = self.__reshape_masses(masses)
        # M = tf.Variable(masses, dtype=tf.float32)
        # M = tf.tile(M, [self._mask_reps])
        # M = tf.reshape(M, (-1, 1))
        # M = tf.boolean_mask(M, self._mask, axis=0)
        # self._M = tf.reshape(M, (self._mask_reps, -1, 1))


    def __reshape_masses(self, masses: np.ndarray):
        M = tf.Variable(masses, dtype=tf.float32)
        M = tf.tile(M, [self._mask_reps])
        M = tf.reshape(M, (-1, 1))
        M = tf.boolean_mask(M, self._mask, axis=0)
        return tf.reshape(M, (self._mask_reps, -1, 1))


    @tf.function
    def _pairwise_distances(self, X: tf.Tensor) -> tf.Tensor:
        """Helper function to calculate the pairwise distances of the locations matrix.
        """
        ## TODO: Could be possibly further optimized. The current solution calculates all N^2 distances, but N^2/2 would be sufficient when switching the signs correctly.
        return tf.expand_dims(X, axis=0) - tf.expand_dims(X, axis=1)


    @tf.function
    def _acceleration(self, Q) -> tf.Tensor:

        ## Calculating Pairwise Distances
        D = self._pairwise_distances(Q[:, :3])

        ## Masking and reshaping X and M to drop self-differences:
        D = tf.reshape(D, (-1, 3))
        D = tf.boolean_mask(D, self._mask, axis=0)
        D = tf.reshape(D, (self._mask_reps, (self._mask_reps-1), -1))

        ## Calculating |xi-xj|^(-3)
        d_inv_cube = tf.pow(tf.reduce_sum(D*D, axis=-1, keepdims=True) + self.smooth, -3./2.) # 1e-20 to smooth numerical overflow
        
        ## Calculating pairwise Force
        F = self._M * D * d_inv_cube

        ## Summation over each other body
        F = self._G * tf.reduce_sum(F, axis=1)

        ## Combining to R6 acceleration
        dQ = tf.concat([Q[:, 3:], F], axis=-1)
        
        return dQ


    @tf.function
    def _solver_rkf(self, Q, f):
        dt = self.dt
        k1 = f(Q)
        k2 = f(Q + dt * k1 / 4.)
        k3 = f(Q + dt * k1 * 3. / 32.      + dt * k2 * 9. / 32.)
        k4 = f(Q + dt * k1 * 1932. / 2197. - dt * k2 * 7200. / 2197. + dt * k3 * 7296. / 2197.)
        k5 = f(Q + dt * k1 * 439. / 216.   - dt * k2 * 8.            + dt * k3 * 3680. / 513   - dt * k4 * 845. / 4104.)
        k6 = f(Q - dt * k1 * 8. / 27.      + dt * k2 * 2.            - dt * k3 * 3544. / 2565. + dt * k4 * 1859. / 4104. - dt * k5 * 11. / 40.)
        Q = Q + dt * (16. / 135. * k1 + 6656. / 12825. * k3 + 28561. / 56430. * k4 - 9. / 50. * k5 + 2. / 55. * k6)
        return Q


    def step(self) -> None:
        Q = self._solver_rkf(self._Q, self._acceleration)
        self._Q = Q
        self._Q_hist = tf.concat([self._Q_hist, [Q]], axis=0)

    
    def simulation(self, steps) -> None:
        for k in tqdm(range(steps)):
            self.step()
        sleep(0.3)


    def plot_history_3d(self) -> None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for n in range(self._Q_hist.shape[1]):
            ax.plot(
                self._Q_hist[:, n, 0], 
                self._Q_hist[:, n, 1], 
                self._Q_hist[:, n, 2]
            )
    
    def plot_history_2d(self, ZSIZE=False, SUBSET=True, n_sample=100) -> None:

        ## TODO: Implement sampler and forcing ZSIZE=False for large scales

        def plot_single_2d(n, ZSIZE=ZSIZE):
            if ZSIZE:
                plt.scatter(
                    self._Q_hist[:, n, 0], 
                    self._Q_hist[:, n, 1], 
                    0.75 * np.clip(self._Q_hist[:, n, 2].numpy(), a_min=1e-10, a_max=np.inf)
                )
            else: 
                plt.scatter(
                    self._Q_hist[:, n, 0], 
                    self._Q_hist[:, n, 1]
                )
        
        plt.figure(figsize=(10, 10))
        if self._Q_hist.shape[1] > 25 and SUBSET==True:
            sample = np.random.choice(np.arange(self._Q_hist.shape[1]), size=n_sample, replace=False)
            for n in sample:
                plot_single_2d(n)
        
        else:
            for n in range(self._Q_hist.shape[1]):
                plot_single_2d(n)

        plt.show()