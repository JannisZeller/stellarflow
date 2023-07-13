import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from abc import ABC, abstractmethod

from .sunsystem import SunSystem
from .walkers import Walker
from .solver import Solver




class ActionHandler(ABC):
    @abstractmethod
    def action_to_boost(self, action):
        pass

    @abstractmethod
    def construct_action_spec(self, max_boost: float):
        pass




class ObservationHandler(ABC):
    @abstractmethod
    def construct_observation_spec(self):
        pass

    @abstractmethod
    def set_current_state(self, is_initial=False):
        if is_initial:
            self._initial_state = self.state




class WalkerSystemEnv(ActionHandler, ObservationHandler, py_environment.PyEnvironment):

    def __init__(self,
        walker: Walker,
        system :SunSystem,
        solver: Solver,
        target: tf.Tensor,
        step_size: float=1.0,
        reward_factor_boost: float=1.0,
        reward_factor_target_distance: float=1.0,
        max_iters: int=1000,
        max_boost: float=1e-4,
    ):

        self.store_main_components(system, solver, walker, target)
        self.set_time_step_size(step_size)
        self.set_reward_factors(
            reward_factor_boost,
            reward_factor_target_distance
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


    def set_reward_factors(
            self,
            reward_factor_boost,
            reward_factor_target_distance
        ):
        self.reward_factor_boost = reward_factor_boost
        self.reward_factor_target_distance = reward_factor_target_distance


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


    def compute_reward(self):
        distance = np.linalg.norm(self.target - self.walker.state_vector[:3])
        boost = self.current_boost[3:]
        reward = (
            - self.reward_factor_boost * np.linalg.norm(boost)
            - self.reward_factor_target_distance * distance
        )
        return reward




class ContinuousAction(ActionHandler):

    def construct_action_spec(self, max_boost):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=self.walker.state_vector.numpy().dtype,
            minimum=-max_boost,  # TODO: Interplay with Mass
            maximum= max_boost,  # TODO: Interplay with Mass
            name="boost"
        )

    def action_to_boost(self, action):
        boost = tf.constant(action, dtype=self.current_boost.dtype)
        self.current_boost[3:].assign(boost)




class DiscreteAction(ActionHandler):

    def construct_action_spec(self, max_boost):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=int,
            minimum=0, maximum=6,
            name="boost-direction"
        )
        self.available_boosts = (
            max_boost * # TODO: Interplay with Mass
            np.array(
                [[ 1.,  0.,  0.],
                 [ 0.,  1.,  0.],
                 [ 0.,  0.,  1.],
                 [-1.,  0.,  0.],
                 [ 0., -1.,  0.],
                 [ 0.,  0., -1.],
                 [ 0.,  0.,  0.]],
                dtype=np.float32)
            )

    def action_to_boost(self, action):
        boost = tf.constant(
            self.available_boosts[action],
            dtype=self.current_boost.dtype
        )
        self.current_boost[3:].assign(boost)




class StateAndDiffObservation(ObservationHandler):

    def construct_observation_spec(self):
        self._observation_spec = {
            "walker-state": array_spec.ArraySpec(
                self.walker.state_vector.shape,
                dtype=self.walker.state_vector.numpy().dtype,
                name="walker-state"
            ),
            "diff-to-target": array_spec.ArraySpec(
                self.target.shape,
                dtype=self.target.numpy().dtype,
                name="target"
            )
        }

    def set_current_state(self, is_initial=False):
        diff_to_target = self.target - self.walker.state_vector[:3]
        self.state = {
            "walker-state": self.walker.state_vector.numpy(),
            "diff-to-target": diff_to_target.numpy()
        }
        super().set_current_state(is_initial)





class GravityObservation(ObservationHandler):

    def construct_observation_spec(self):
        self._observation_spec = {
            "gravity": array_spec.ArraySpec(
                self.walker.state_vector[3:].shape,
                dtype=self.walker.state_vector.numpy().dtype,
                name="gravity"
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

    def set_current_state(self, is_initial=False):
        gravity = self.walker.compute_acceleration(
            self.walker.state_vector
        )
        self.state = {
            "gravity": gravity[3:].numpy(),
            "walker-state": self.walker.state_vector.numpy(),
            "target": self.target.numpy()
        }
        super().set_current_state(is_initial)




class AllPositionsObservation(ObservationHandler):

    def construct_observation_spec(self):
        self._observation_spec = {
            "system-positions": array_spec.ArraySpec(
                self.system.positions_tensor.shape,
                dtype=self.system.positions_tensor.numpy().dtype,
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

    def set_current_state(self, is_initial=False):
        self.state = {
            "system-positions": self.system.positions_tensor.numpy(),
            "walker-state": self.walker.state_vector.numpy(),
            "target": self.target.numpy()
        }
        super().set_current_state(is_initial)
