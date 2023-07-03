import tensorflow as tf
from typing import Callable


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
