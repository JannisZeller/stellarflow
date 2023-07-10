import numpy as np
from astropy.coordinates import CartesianRepresentation
from dataclasses import dataclass


@dataclass
class Positions():
    """
    Wrapper for convenient getting positions of multiple bodies that are given
    in `astropy.coordinates.CartesianRepresentation`s.
    """

    data: np.ndarray

    def __init__(self, bodies_cartesians: list[CartesianRepresentation]):
        self.data = np.vstack([np.array(body.xyz) for body in bodies_cartesians])

    def get_data(self):
        return self.data
