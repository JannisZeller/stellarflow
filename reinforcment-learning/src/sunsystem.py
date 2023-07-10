import tensorflow as tf # noqa: F401


class SunSystem():
    """
    A base class for describing the sun system as a system of massive bodies.
    Provides a gravitational field for calculating accellearation of "test"-
    bodies at different locations.
    """
    # Gravitational constant for length in astronomical units, mass in
    #   sun-masses and time in earth days.
    gravitational_constant: float = 0.0002959211565456235

    def __init__(self):
        pass


    def __repr__(self):
        return f"SunSystem" # noqa: F541

    def __str__(self):
        return f"SunSystem"  # noqa: F541
