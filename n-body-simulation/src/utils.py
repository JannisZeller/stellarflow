import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

# from . import n_body_system


def equations_of_motion_solver_factory(
    algorithm: str = "rk4", step_size: float = 1.0
) -> Callable:
    """Returns a Runge-Kutta-style solver for integrating dQ = dQ(Q).
    Q represents the state matrix (x, v) with positions x and velocities v and
    dQ represents the states matrix' "derivative", i. e. dQ = (v, a) where
    a is the acceleration.
    The factory pattern is used to bind a step size to the function despite
    using the `@tf.function` wrapper.
    """
    if algorithm == "rk4":

        @tf.function
        def solver_rk4(Q: tf.Tensor, dQ: Callable):
            """Runge-Kutta solver."""
            k1 = dQ(Q)
            k2 = dQ(Q + step_size * k1 / 2.0)
            k3 = dQ(Q + step_size * k2 / 2.0)
            k4 = dQ(Q + step_size * k3)
            return Q + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        return solver_rk4

    elif algorithm == "rkf":

        @tf.function
        def solver_rkf(Q: tf.Tensor, dQ: Callable):
            """Runge-Kutta-Fehlberg solver."""
            k1 = dQ(Q)
            k2 = dQ(Q + step_size * k1 / 4.0)
            k3 = dQ(Q + step_size * k1 * 3.0 / 32.0 + step_size * k2 * 9.0 / 32.0)
            k4 = dQ(
                Q
                + step_size * k1 * 1932.0 / 2197.0
                - step_size * k2 * 7200.0 / 2197.0
                + step_size * k3 * 7296.0 / 2197.0
            )
            k5 = dQ(
                Q
                + step_size * k1 * 439.0 / 216.0
                - step_size * k2 * 8.0
                + step_size * k3 * 3680.0 / 513
                - step_size * k4 * 845.0 / 4104.0
            )
            k6 = dQ(
                Q
                - step_size * k1 * 8.0 / 27.0
                + step_size * k2 * 2.0
                - step_size * k3 * 3544.0 / 2565.0
                + step_size * k4 * 1859.0 / 4104.0
                - step_size * k5 * 11.0 / 40.0
            )
            Q = Q + step_size * (
                16.0 / 135.0 * k1
                + 6656.0 / 12825.0 * k3
                + 28561.0 / 56430.0 * k4
                - 9.0 / 50.0 * k5
                + 2.0 / 55.0 * k6
            )
            return Q

        return solver_rkf

    else:
        raise ValueError('`algorithm` must be of ["rk4", "rkf"]')


def plot_system_history(system, mode: str = "3d", **kwargs):    # : n_body_system.NBodySystem
    """A plotting utility to plot an NBodySystem."""

    if mode not in ["3d", "2d"]:
        raise ValueError('`mode` must be of ["3d", "2d"].')

    if 'figsize' in kwargs:
        figsize = kwargs.get('figsize')
    else:
        figsize = (7, 7)

    # Sampling a subset of indices of bodies of potentially very large systems.
    if 'n_sample' in kwargs:
        if system.state_history.shape[1] > kwargs.get('n_sample'):
            sample = np.random.choice(
                np.arange(system.state_history.shape[1]),
                size=kwargs.get('n_sample'),
                replace=False,
            )
    else:
        sample = range(system.state_history.shape[1])

    if mode == "3d":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        for idx in sample:
            ax.plot(
                system.state_history[:, idx, 0],
                system.state_history[:, idx, 1],
                system.state_history[:, idx, 2],
            )

    if mode == "2d":

        def plot_single_2d(n):
            if 'zsize' in kwargs:
                plt.scatter(
                    system.state_history[:, n, 0],
                    system.state_history[:, n, 1],
                    0.75
                    * np.clip(
                        system.state_history[:, n, 2].numpy(), a_min=1e-10, a_max=np.inf
                    ),
                )
            else:
                plt.scatter(
                    system.state_history[:, n, 0], system.state_history[:, n, 1]
                )

        plt.figure(figsize=figsize)
        for idx in sample:
            plot_single_2d(idx)
        plt.show()
