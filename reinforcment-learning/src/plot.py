import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from seaborn import scatterplot
import matplotlib.pyplot as plt

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

    sizes = np.ones(df.shape[0])
    sizes[df["body"] == "sun"] = 5
    df["size"] = sizes

    return df


def generate_plot_df_walker(walker: Walker):
    df = pd.DataFrame(walker.state_history[:, :3].numpy())
    df.columns = ["x", "y", "z"]
    df["body"] = walker.name
    df["size"] = 1
    return df


class Plotter():

    def __init__(self, **kwargs):
        if "system" in kwargs and "walker" in kwargs:
            system = kwargs.get("system")
            walker = kwargs.get("walker")
            self.df = generate_plot_df(system, walker)
            self.n_bodies = system.n_bodies + 1
        elif "walker" in kwargs:
            self.df = generate_plot_df_walker(kwargs.get("walker"))
            self.n_bodies = 1
        elif "system" in kwargs:
            system = kwargs.get("system")
            self.df = generate_plot_df_system()
            self.n_bodies = system.n_bodies


    def draw(self, mode: str="2d"):
        if mode == "2d":
            self.plot2d()
        if mode == "3d":
            self.plot3d()


    def plot3d(self):
        fig1 = px.scatter_3d(
            self.df[self.df["body"]=="sun"], x="x", y="y", z="z", color="body",
            size="size", color_discrete_sequence=["orange"]
        )
        fig2 = px.line_3d(
            self.df[self.df["body"]!="sun"], x="x", y="y", z="z", color="body",
        )
        fig3 = go.Figure(data=fig1.data + fig2.data)
        fig3.update_layout(
            scene = {
                'zaxis': {
                    'nticks': 4,
                    'range': [-1, 1]
                }
            }
        )
        fig3.show()
        return fig3


    def plot2d(self):
        fig = scatterplot(self.df, x="x", y="y", hue="body", size="size")
        handles, leg = fig.get_legend_handles_labels()
        plt.legend(
            handles[0 : self.n_bodies+1],
            leg[0 : self.n_bodies+1],
            loc=2,
            borderaxespad=0.
        )
        plt.show(fig)
        return fig
