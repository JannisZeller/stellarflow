# %% Imports
#-------------------------------------------------------------------------------

import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils

#-------------------------------------------------------------------------------



class stellarEnv(py_environment.PyEnvironment):
    ## TODO: Implement mass / propellant consumtion
    def __init__(self,
            mass: float,
            initial_location: np.ndarray,
            initial_velocity: np.ndarray,
            target_point:   np.ndarray,
            target_point_relative:      bool=False,
            target_point_relative_to:   int=None,
            reward_factor_boost:           float=1.0,
            reward_factor_target_distance: float=1.0):

        ## Additional Information for spacecraft
        self._m = mass
        self._m_hist = [mass]
        self._x = np.array(initial_location)
        self._y = np.array(initial_velocity)
        self._q = tf.constant(
            np.concatenate(
                [initial_location, initial_velocity],
                axis=0
            ),
            dtype=tf.float32
        )
        self._reward_factor_boost = reward_factor_boost
        self._reward_factor_target_distance = reward_factor_target_distance

        ## Target Location for rewards
        self._target = tf.constant(target_point, dtype=tf.float32)
        self._target_point_relative = target_point_relative
        if target_point_relative and target_point_relative_to is None:
            print("No target point reference object ('target_point_relative_to') provided, using first object of System.")
            self._target_point_relative_to = 0
        else:
            self._target_point_relative_to = target_point_relative_to

        ## Binding with stf.System:
        stfSystem._Q = tf.Variable(
            np.concatenate(
                [stfSystem._Q.numpy(), np.expand_dims(self._q, axis=0)],
                axis=0),
            dtype=tf.float32
        )
        stfSystem._Q_hist = tf.cast(tf.expand_dims(stfSystem._Q, axis=0), dtype=tf.float32)
        stfSystem._mask_reps, stfSystem._mask_len, stfSystem._mask = stfSystem._create_mask(stfSystem._Q.shape[0])
        stfSystem.masses = np.concatenate([stfSystem.masses, [mass]], axis=0) # Maybe put mass to 0
        stfSystem._M = stfSystem._reshape_masses(stfSystem.masses)
        self._sys = stfSystem

        ## State for max iterations
        self.n_iter = 0

        ## Providing for tfa-py_env
        #  TODO: Better values for minimum and maximum
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=np.float32,
            minimum=-1e-31,
            maximum=1e-31,
            name="boost"
            )
        self._observation_spec = {
            "q-vector": array_spec.ArraySpec(self._sys._Q.shape, dtype=np.float32, name="q-vector"),
            "target": array_spec.ArraySpec((3,), dtype=np.float32, name="target")
        }
        self._state = self._sys._Q.numpy()
        self._initial_q = self._sys._Q
        self._episode_ended = False

        ## Variable for boost (Defy tf's-Error: "AssertionError: Called a function referencing variables which have been deleted.")
        self._current_boost = tf.Variable(
            tf.zeros_like(self._sys._Q, dtype=self._sys._Q.dtype)
        )


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = {"q-vector": self._initial_q.numpy(), "target": self._target.numpy()}
        self._episode_ended = False
        self._sys._reset()
        return ts.restart(self._state)

    def _compute_target_distance(self):
        # TODO: Target point might need to be a function of steps, which needs to get hand designed for flexibility (e. g. Lagrange Points).
        if self._target_point_relative:
            target = self._target + self._sys._Q[self._target_point_relative_to, :3]
            return target - self._q[:3]
        else:
            return self._target - self._q[:3]

    # def _set_new_mass(self):
    #     # TODO: Effect of boosting (propellant consumption)
    #     self._m = self._m
    #     self._m_hist = np.concatenate([self._m_hist, [self._m]], axis=0)

    @tf.function
    def _accelerations_plus_boost(self, Q):
        return self._sys._acceleration(Q) + self._current_boost / self._m

    def _step(self, action: np.ndarray):
        self.n_iter += 1

        if self._episode_ended:
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
            return self.reset()

        # Defining boost
        self._current_boost[-1, 3:].assign(
            tf.constant(action, dtype=self._current_boost.dtype)
        )

        # Performing a system step
        Q = self._sys._Q
        Q = self._sys._solver_rkf(Q, self._accelerations_plus_boost)
        self._sys._Q_hist = tf.concat([self._sys._Q_hist, [Q]], axis=0)
        self._sys._Q = Q
        self._q = Q[-1]
        # self._set_new_mass()
        self._state = {"q-vector": self._sys._Q.numpy(), "target": self._target.numpy()}

        # Bounding episode length
        if self._episode_ended or self.n_iter > 1000:
            reward = 0.0
            return ts.termination(self._state, reward)

        # Rewarding with boost length and distance to target
        else:
            distance = np.linalg.norm(self._compute_target_distance())
            target = self._sys._Q[self._target_point_relative_to]
            reward = -self._reward_factor_boost*np.linalg.norm(action) - self._reward_factor_target_distance*distance
            return ts.transition(
                self._state, reward=reward, discount=1.0
            )


if __name__ == "__main__":

    import sys
    if "..\\" not in sys.path: sys.path.append("..\\")


    AU, ED = nBodySystem._AU, nBodySystem._ED

    X = np.array([
        [0., 0.,    0.], # Sun
        [1., 0.,    0.], # Earth
        [0., 1.524, 0.]  # Mars
    ])

    V = np.array([
        [0.,           0., 0.],  # Sun
        [0., 29290./AU*ED, 0.],  # Earth
        [27070./AU*ED, 0., 0.],  # Mars
    ])

    M = np.array([
        1.,                # Sun
        3.0025e-6,         # Earth
        0.107 * 3.0025e-6  # Mars
    ])

    SEM_system = nBodySystem(X, V, M, smooth=1e-10)

    env = stellarEnv(mass=1e-28, initial_location=[1.1, 0., 0.], initial_velocity=[0., 0.01, 0.], target_point=[2., 0., 0.], stfSystem=SEM_system)
    utils.validate_py_environment(env, episodes=5)

    print("Validated as python tensorflow-agents environment")
