import numpy as np

from abc import ABC, abstractmethod


class Target(ABC):

    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def state_history(self):
        pass

    @abstractmethod
    def propagate(self):
        pass

    @abstractmethod
    def get_location(self):
        pass

    @abstractmethod
    def reset(self):
        pass




class FixedTarget(Target):

    def __init__(self, name: str, location: np.ndarray):
        super().__init__(name)

        self.state_vector = np.zeros((6,))
        self.state_vector[:3] = location
        self.state_history = np.expand_dims(self.state_vector, axis=0)

    def propagate(self):
        self.state_history = np.concatenate(
            [self.state_history, [self.state_vector]]
        )

    def get_location(self):
        return self.state_vector[:3]

    def reset(self):
        self.state_vector = self.state_history[0, ...]
        self.state_history = np.expand_dims(self.state_vector, axis=0)
