import numpy as np
import tensorflow as tf

from tf_agents.specs import array_spec

from abc import abstractmethod

from ._base import BaseHandler



class ActionHandler(BaseHandler):

    current_boost: tf.Tensor

    @abstractmethod
    def action_to_boost(self, action):
        pass


    @abstractmethod
    def construct_action_spec(self, max_boost: float):
        self.max_boost = max_boost




class ContinuousAction(ActionHandler):

    def construct_action_spec(self, max_boost: float):
        super().construct_action_spec(max_boost)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=self.walker.state_vector_numpy.dtype,
            minimum=-max_boost,  # TODO: Interplay with Mass
            maximum= max_boost,  # TODO: Interplay with Mass
            name="boost"
        )


    def action_to_boost(self, action):
        boost = tf.constant(action, dtype=self.current_boost.dtype)
        boost = tf.clip_by_norm(boost, self.max_boost)
        self.current_boost[3:].assign(boost)



class DiscreteAction(ActionHandler):

    def construct_action_spec(self, max_boost):
        super().construct_action_spec(max_boost)
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



class OneDimDiscreteAction(DiscreteAction):

    def construct_action_spec(self, max_boost):
        super().construct_action_spec(max_boost)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=int,
            minimum=0, maximum=2,
            name="boost-direction"
        )
        self.available_boosts = (
            max_boost * # TODO: Interplay with Mass
            np.array(
                [[ 0.,  0.,  1.],
                 [ 0.,  0., -1.],
                 [ 0.,  0.,  0.]],
                dtype=np.float32)
            )
