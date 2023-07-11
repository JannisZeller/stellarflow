import numpy as np
from astropy.time import Time as AstroTime
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import get_body_barycentric_posvel

from .config import config


def get_x_rotation_matrix(theta: float=23.45*np.pi / 180.):
    x_rotation_matrix = np.array(
        [[1,             0,              0],
        [0,  np.cos(theta), np.sin(theta)],
        [0, -np.sin(theta), np.cos(theta)]]
    )
    return x_rotation_matrix


def get_cartesians(
        bodies: list[str],
        time: AstroTime,
        idx: int
    ) -> dict[str, CartesianRepresentation]:
    """Returns a dict of `CartesianRepresentations` for the bodies passed in.
    The bodies have to be available in Astropy's `get_body_barycentric_posvel`.
    The `idx` determines wether positions (0) or velocities (1) are returned.
    """
    cartesians = {}
    for body in bodies:
        cartesians[body] = get_body_barycentric_posvel(body, time)[idx] # Returns AU and AU / EarthDay                  # noqa: E501
    return cartesians


def get_cartesian_positions(
        bodies: list[str],
        time: AstroTime,
    ) -> dict[str, CartesianRepresentation]:
    return get_cartesians(bodies, time, 0)


def get_cartesian_velocities(
        bodies: list[str],
        time: AstroTime,
    ) -> dict[str, CartesianRepresentation]:
    return get_cartesians(bodies, time, 1)


def cartesians_to_array(
        cartesians: dict[str, CartesianRepresentation],
    ) -> np.ndarray:
    """Transforms a cartesians-dict to a 2d array with the positions stacked
    vertiacally (i. e. shape `n_bodies x 3`).
    The data generated by the astropy-package is rotated approx. 23.45° around
    the x-axis compared to the orbital plane of most planets. If the
    `rotate_to_sun_earth_plane` is true, the vectors get rotated to lie in
    the plane again
    (see https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions).
    """
    array_list = []
    for cartesian in cartesians.values():
        cartesian_as_array = np.array(cartesian.xyz)
        if config.rotate_to_sun_earth_plane is True:
            x_rotation_matrix = get_x_rotation_matrix()
            array_list.append(cartesian_as_array @ x_rotation_matrix.T)
        else:
            array_list.append(cartesian_as_array)
    return np.vstack(array_list)
