import numpy as np
import pandas as pd

from ..components import SunSystem, Walker, Target
from ..environment import WalkerSystemEnv


_body_colname = "body"
_position_colnames = ["x", "y", "z"]


def generate_plot_data_env(env: WalkerSystemEnv):
    df_system = generate_plot_data_system(env.system)
    df_target = generate_plot_data_target(env.target)
    df_walker = generate_plot_data_walker(env.walker)
    return pd.concat([df_system, df_walker, df_target], axis=0)


def generate_plot_data_system(system: SunSystem):
    n_datapoints = system.positions_history.shape[0]
    n_bodies = system.n_bodies

    history_2d = system.positions_history.numpy().reshape(-1, 3)
    df = pd.DataFrame(history_2d)
    df.columns = _position_colnames

    names = np.empty((n_bodies * n_datapoints,), object)
    for n in range(n_bodies):
        names[n::n_bodies] = system.bodies[n]

    df[_body_colname] = names

    sizes = np.ones(df.shape[0])
    sizes[df[_body_colname] == "sun"] = 5
    df["size"] = sizes

    return df


def generate_plot_data_walker(walker: Walker):
    df = pd.DataFrame(walker.state_history[:, :3].numpy())
    df.columns = _position_colnames
    df[_body_colname] = walker.name
    df["size"] = 1
    return df


def generate_plot_data_target(target: Target):
    df = pd.DataFrame(target.state_history[:, :3])
    df.columns = _position_colnames
    df[_body_colname] = "target"
    df["size"] = 1
    return df
