import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .sunsystem import SunSystem
from .walkers import Walker
from .solver import Solver



class WalkerSystemEnv(py_environment.PyEnvironment):
    """
    https://towardsdatascience.com/creating-a-custom-environment-for-tensorflow-agent-tic-tac-toe-example-b66902f73059
    """
    def __init__(self,
            walker: Walker,
            reference_system :SunSystem,
            solver: Solver,
            target: np.ndarray,
            step_size: float=1.0,
            reward_factor_boost: float=1.0,
            reward_factor_target_distance: float=1.0,
            max_iters: int=1000):

        self.walker = walker
        if not hasattr(walker, "reference_system"):
            walker.set_reference_system(reference_system)
        self.reference_system = reference_system
        self.solver = solver

        self.step_size = step_size
        self.set_time_step_size()

        self.reward_factor_boost = reward_factor_boost
        self.reward_factor_target_distance = reward_factor_target_distance

        self.target = tf.constant(target, dtype=tf.float32)

        self.n_iter = 0
        self.max_iters = max_iters

        ## Providing for tfa-py_env
        #  TODO: Better values for minimum and maximum
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=self.walker.state_vector.numpy().dtype,
            minimum=-0.0001,  # TODO: Interplay with Mass
            maximum= 0.0001,  # TODO: Interplay with Mass
            name="boost"
        )
        self._observation_spec = {
            "system-positions": array_spec.ArraySpec(
                self.reference_system.positions_tensor.shape,
                dtype=self.reference_system.positions_tensor.numpy().dtype,
                name="system-positions"
            ),
            "walker-state": array_spec.ArraySpec(
                self.walker.state_vector.shape,
                dtype=self.walker.state_vector.numpy().dtype,
                name="walker-state"
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


    def set_time_step_size(self):
        self.reference_system.step_size = self.step_size
        self.solver.step_size = self.step_size


    def _reset(self):
        self._episode_ended = False
        self.reference_system._reset()
        self.walker._reset()
        self._state = {
            "system-positions": self.reference_system.positions_tensor.numpy(),
            "walker-state": self.walker.state_vector.numpy(),
            "target": self.target.numpy()
        }
        self.n_iter = 0
        # print("Resetted WalkerSystemEnv.")
        return ts.restart(self._state)


    def compute_target_distance(self):
        # TODO: Target point might need to be a function of steps, which
        #   needs to get hand designed for flexibility (e. g. Lagrange Points).
        return np.linalg.norm(self.target - self.walker.state_vector[:3])


    # def _set_new_mass(self):
    #     # TODO: Effect of boosting (propellant consumption)
    #     self._m = self._m
    #     self._m_hist = np.concatenate([self._m_hist, [self._m]], axis=0)


    @tf.function
    def walker_accelerations_plus_boost(self, state_vector):
        gravitation = self.walker.compute_acceleration(state_vector)
        boost = self._current_boost / self.walker.mass
        return gravitation + boost  # TODO: Interplay with action bounds


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
            self.walker_accelerations_plus_boost
        )
        self.walker.state_history = tf.concat(
            [self.walker.state_history, [self.walker.state_vector]],
            axis=0
        )
        self.reference_system.propagate()

        # if self.n_iter % (self.max_iters // 10) == 0:
        #     print(f"Iter: {self.n_iter} -- Date: {self.reference_system.time}")


        self._state = {
            "system-positions": self.reference_system.positions_tensor.numpy(),
            "walker-state": self.walker.state_vector.numpy(),
            "target": self.target.numpy()
        }

        distance = self.compute_target_distance()
        reward = (
            - self.reward_factor_boost * np.linalg.norm(action)
            - self.reward_factor_target_distance * distance
        )

        # Bounding episode length
        if self._episode_ended or self.n_iter >= self.max_iters:
            return ts.termination(self._state, reward)

        # Rewarding with boost length and distance to target
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)
