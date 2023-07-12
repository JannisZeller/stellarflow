import tensorflow as tf
from typing import Callable


class Solver():
    """
    Runge-Kutta-style solver for integrating dQ = dQ(Q).
    Q represents the state matrix (x, v) with positions x and velocities v and
    dQ represents the states matrix' "derivative", i. e. dQ = (v, a) where
    a is the acceleration.
    """
    def __init__(self, algorithm: str = "rk4", step_size: float = 1.0):
        if algorithm not in ["rk4", "rkf"]:
            raise ValueError('`algorithm` must be of ["rk4", "rkf"]')
        self.step_size = step_size
        self.algorithm = algorithm


    def __call__(self, Q: tf.Tensor, dQ: Callable):
        if self.algorithm == "rk4":
            return self.solver_rk4(Q, dQ)
        elif self.algorithm == "rkf":
            return self.solver_rkf(Q, dQ)


    @tf.function
    def solver_rk4(self, Q: tf.Tensor, dQ: Callable):
        """Runge-Kutta solver."""
        dt = self.step_size
        k1 = dQ(Q)
        k2 = dQ(Q + dt * k1 / 2.0)
        k3 = dQ(Q + dt * k2 / 2.0)
        k4 = dQ(Q + dt * k3)
        return Q + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


    @tf.function
    def solver_rkf(self, Q: tf.Tensor, dQ: Callable):
        """Runge-Kutta-Fehlberg solver."""
        dt = self.step_size
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
