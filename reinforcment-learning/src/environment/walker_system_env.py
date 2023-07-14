import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


from .action_handlers import ActionHandler
from .observation_handlers import ObservationHandler
from .reward_handlers import RewardHanlder

from ..components import Walker
from ..components import SunSystem
from ..components import Solver


class WalkerSystemEnv(
    ActionHandler,
    ObservationHandler,
    RewardHanlder,
    py_environment.PyEnvironment
    ):

    def __init__(self,
        walker: Walker,
        system: SunSystem,
        solver: Solver,
        target: tf.Tensor,
        step_size: float=1.0,
        reward_factor_boost: float=-1.0,
        reward_factor_target_distance: float=-1.0,
        reward_per_step: float=-1.0,
        reward_per_step_near_target: float=1.,
        near_target_window: float=0.01,
        max_iters: int=1000,
        max_boost: float=5e-5,
    ):

        self.store_main_components(system, solver, walker, target)
        self.set_time_step_size(step_size)
        self.set_reward_factors(
            reward_factor_boost,
            reward_factor_target_distance,
            reward_per_step,
            reward_per_step_near_target,
            near_target_window
        )
        self.handle_iterations(max_iters)

        self.construct_action_spec(max_boost)
        self.construct_observation_spec()

        self.set_current_state(is_initial=True)

        self._episode_ended = False

        # Variable for boost (Defy tf's-Error: "AssertionError: Called a
        #   function referencing variables which have been deleted.")
        self.current_boost = tf.Variable(
            tf.zeros_like(
                self.walker.state_vector,
                dtype=self.walker.state_vector.dtype
        ))


    def _step(self, action: np.ndarray):
        if self._episode_ended:
            return self.reset()
        self.n_iter += 1

        self.action_to_boost(action)

        self.propagate_system_walker()

        self.set_current_state()

        reward = self.compute_reward()

        # Bounding episode length
        if self._episode_ended or self.n_iter >= self.max_iters:
            return ts.termination(self.state, reward)

        # Rewarding with boost length and distance to target
        else:
            return ts.transition(self.state, reward=reward, discount=1.0)


    def action_spec(self):
        return self._action_spec


    def observation_spec(self):
        return self._observation_spec


    def _reset(self):
        self._episode_ended = False
        self.system._reset()
        self.walker._reset()
        self.state = self._initial_state
        self.n_iter = 0
        # print("Resetted WalkerSystemEnv.")
        return ts.restart(self.state)


    def store_main_components(self, system, solver, walker, target):
        self.walker = walker
        if not hasattr(walker, "system"):
            walker.set_reference_system(system)
        self.system = system
        self.solver = solver
        self.target = target


    def set_time_step_size(self, step_size):
        self.step_size = step_size
        self.system.step_size = step_size
        self.solver.step_size = step_size


    def handle_iterations(self, max_iters):
        self.n_iter = 0
        self.max_iters = max_iters


    @tf.function
    def walker_accelerations_plus_boost(self, state_vector):
        gravitation = self.walker.compute_acceleration(state_vector)
        boost = self.current_boost / self.walker.mass
        return gravitation + boost  # TODO: Interplay with action bounds


    def propagate_system_walker(self):
        self.walker.state_vector = self.solver(
            self.walker.state_vector,
            self.walker_accelerations_plus_boost
        )
        self.walker.state_history = tf.concat(
            [self.walker.state_history, [self.walker.state_vector]],
            axis=0
        )
        self.system.propagate()
