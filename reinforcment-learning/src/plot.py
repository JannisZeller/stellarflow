import numpy as np
import pandas as pd
import plotly.express as px
from seaborn import scatterplot

from .sunsystem import SunSystem
from .walkers import Walker


def generate_plot_df(system: SunSystem, walker: Walker) ->pd.DataFrame:
    """Generates a dataframe for a system or a walker for plotting purposes.
    """
    df_system = generate_plot_df_system(system)
    df_walker = generate_plot_df_walker(walker)
    return pd.concat([df_system, df_walker], axis=0)


def generate_plot_df_system(system: SunSystem):
    n_datapoints = system.positions_history.shape[0]
    n_bodies = system.n_bodies

    history_2d = system.positions_history.numpy().reshape(-1, 3)
    df = pd.DataFrame(history_2d)
    df.columns = ["x", "y", "z"]

    names = np.empty((n_bodies * n_datapoints,), object)
    for n in range(n_bodies):
        names[n::n_bodies] = system.bodies[n]

    df["body"] = names

    return df


def generate_plot_df_walker(walker: Walker):
    df = pd.DataFrame(walker.state_history[:, :3].numpy())
    df.columns = ["x", "y", "z"]
    df["body"] = walker.name
    return df


class Plotter():

    def __init__(self, **kwargs):
        if "system" in kwargs and "walker" in kwargs:
            self.df = generate_plot_df(kwargs.get("system"), kwargs.get("walker"))
        elif "walker" in kwargs:
            self.df = generate_plot_df_walker(kwargs.get("walker"))
        elif "system" in kwargs:
            self.df = generate_plot_df_system(kwargs.get("system"))


    def draw(self, mode: str="2d"):
        if mode == "2d":
            self.plot2d()
        if mode == "3d":
            self.plot3d()


    def plot3d(self):
        fig = px.line_3d(self.df, x="x", y="y", z="z", color="body")
        fig.update_layout(
            scene = dict(zaxis = dict(nticks=4, range=[-1,1]))
        )
        fig.show()
        return fig


    def plot2d(self):
        fig = scatterplot(self.df, x="x", y="y", hue="body")
        return fig
