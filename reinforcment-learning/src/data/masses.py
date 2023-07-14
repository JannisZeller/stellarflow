
sun_mass_kg = 1.98840987e+30    # sun mass in kg
au_m = 1.49597871e+11           # astronomical unit in meters
day_s = 86400.                  # earth day in seconds

bodies_masses = {
    'earth':    5.9722e+24 / sun_mass_kg,
    'moon':     7.3420e+22 / sun_mass_kg,
    'mars':     6.4171e+23 / sun_mass_kg,
    'venus':    4.8670e+24 / sun_mass_kg,
    'sun':      1.,
    'jupiter':  1.8981246e+27 / sun_mass_kg,
    'neptune':  1.024e+26 / sun_mass_kg
}


def get_masses(bodies: list[str]):
    return [bodies_masses[body] for body in bodies]
