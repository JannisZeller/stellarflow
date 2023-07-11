import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
# from tf_agents.environments import utils

from typing import Callable

from .sunsystem import SunSystem
from .walkers import Walker



class WalkerSystemEnv(py_environment.PyEnvironment):
    """
    https://towardsdatascience.com/creating-a-custom-environment-for-tensorflow-agent-tic-tac-toe-example-b66902f73059
    """
    def __init__(self,
            walker: Walker,
            reference_system :SunSystem,
            solver: Callable,
            target: np.ndarray,
            dt: float=1.0,
            reward_factor_boost: float=1.0,
            reward_factor_target_distance: float=1.0):

        self.walker = walker
        self.reference_system = reference_system
        self.solver = solver
        self.dt = dt

        self.reward_factor_boost = reward_factor_boost
        self.reward_factor_target_distance = reward_factor_target_distance

        self.target = tf.constant(target, dtype=tf.float32)
        self.n_iter = 0

        ## Providing for tfa-py_env
        #  TODO: Better values for minimum and maximum
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=self.walker.state_vector.numpy().dtype,
            minimum=-1e-31,
            maximum=1e-31,
            name="boost"
        )
        self._observation_spec = {
            "system-positions": array_spec.ArraySpec(
                self.reference_system.positions_tensor.shape,
                dtype=self.reference_system.positions_tensor.numpy().dtype,
                name="System Positions"
            ),
            "walker-state": array_spec.ArraySpec(
                self.walker.state_vector.shape,
                dtype=self.walker.state_vector.numpy().dtype,
                name="Walker Positions"
            ),
            "target": array_spec.ArraySpec(
                self.target.shape,
                dtype=self.target.numpy().dtype,
                name="target"
            )
        }
        self._state = {
            "system-positions": self.reference_system.positions_tensor.numpy(),
            "walker-state": self.walker.state_vector.numpy(),
            "target": self.target.numpy()
        }

        # For reset
        self._initial_walker_state = self._state
        self._episode_ended = False

        # Variable for boost (Defy tf's-Error: "AssertionError: Called a
        #   function referencing variables which have been deleted.")
        self._current_boost = tf.Variable(
            tf.zeros_like(self.walker.state_vector, dtype=self.walker.state_vector.dtype)  # noqa: E501
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


    def _reset(self):
        self._episode_ended = False
        self.reference_system._reset()
        self.walker._reset()
        self._state = {
            "system-positions": self.reference_system.positions_tensor.numpy(),
            "walker-state": self.walker.state_vector.numpy(),
            "target": self.target.numpy()
        }
        return ts.restart(self._state)


    def _compute_target_distance(self):
        # TODO: Target point might need to be a function of steps, which
        #   needs to get hand designed for flexibility (e. g. Lagrange Points).
        return np.linalg.norm(self.target - self.walker.state_vector[:3])


    # def _set_new_mass(self):
    #     # TODO: Effect of boosting (propellant consumption)
    #     self._m = self._m
    #     self._m_hist = np.concatenate([self._m_hist, [self._m]], axis=0)


    @tf.function
    def _walker_accelerations_plus_boost(self, state_vector):
        gravitation = self.walker.compute_acceleration(state_vector)
        boost = self._current_boost / self.walker.mass
        return gravitation + boost


    def _step(self, action: np.ndarray):

        if self._episode_ended:
            return self.reset()

        self.n_iter += 1

        # Defining boost
        self._current_boost[3:].assign(
            tf.constant(action, dtype=self._current_boost.dtype)
        )

        # Propagating
        self.walker.state_vector = self.solver(
            self.walker.state_vector,
            self._walker_accelerations_plus_boost
        )
        self.walker.state_history = tf.concat(
            [self.walker.state_history, [self.walker.state_vector]],
            axis=0
        )
        self.reference_system.propagate(self.dt)


        self._state = {
            "system-positions": self.reference_system.positions_tensor.numpy(),
            "walker-state": self.walker.state_vector.numpy(),
            "target": self.target.numpy()
        }

        # Bounding episode length
        if self.n_iter > 1000:
            reward = 0.0
            return ts.termination(self._state, reward)

        # Rewarding with boost length and distance to target
        else:
            distance = self._compute_target_distance()
            reward = (
                - self.reward_factor_boost * np.linalg.norm(action)
                - self.reward_factor_target_distance * distance
            )
            return ts.transition(
                self._state, reward=reward, discount=1.0
            )
