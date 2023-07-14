import numpy as np

from abc import ABC

from ..components import Walker
from ..components import Target


class BaseHandler(ABC):

    target: Target
    walker: Walker

    def compute_distance(self):
        target = self.target.position
        position = self.walker.position
        return np.linalg.norm(target - position)
