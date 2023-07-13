import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from seaborn import scatterplot
import matplotlib.pyplot as plt

from .sunsystem import SunSystem
from .walkers import Walker


def generate_plot_df(
        system: SunSystem,
        walker: Walker,
        target: np.ndarray
    ) ->pd.DataFrame:
    """Generates a dataframe for a system or a walker for plotting purposes.
    """
    df_system = generate_plot_df_system(system)
    df_walker = generate_plot_df_walker(walker)
    df_target = generate_plot_df_target(target)
    return pd.concat([df_system, df_walker, df_target], axis=0)


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


def generate_plot_df_target(target: np.ndarray):
    try:
        target = target.numpy()
    except AttributeError:
        pass
    df = pd.DataFrame(target.reshape(1, -1))
    df.columns = ["x", "y", "z"]
    df["body"] = "target"
    df["size"] = 1
    return df


class Plotter:

    n_bodies = 0

    def __init__(self, **kwargs):
        if "walker" in kwargs:
            walker = kwargs.get("walker")
            df_walker = generate_plot_df_walker(walker)
            self.df = df_walker
            self.n_bodies += 1

        if "system" in kwargs:
            system = kwargs.get("system")
            df_system = generate_plot_df_system(system)
            self.n_bodies += system.n_bodies
            self.df = pd.concat([self.df, df_system], axis=0)

        if "target" in kwargs:
            target = kwargs.get("target")
            df_target = generate_plot_df_target(target)
            self.n_bodies += system.n_bodies
            self.df = pd.concat([self.df, df_target], axis=0)

        if "env" in kwargs:
            env = kwargs.get("env")
            system = env.system
            walker = env.walker
            target = env.target
            self.df = generate_plot_df(system, walker, target)
            self.n_bodies = system.n_bodies + 2

        if not hasattr(self, "df"):
            raise ValueError(
                "At least pass one thing to plot. " +
                "Typo with the kwarg-names?"
            )


    def draw(self, mode: str="2d", **kwargs):
        if mode == "2d":
            self.plot2d(**kwargs)
        if mode == "3d":
            self.plot3d(**kwargs)


    def plot3d(self, zrange: list[int]=[-3, 3]):
        fig_sun = px.scatter_3d(
            self.df[self.df["body"] == "sun"],
            x="x", y="y", z="z", color="body",
            size="size",
            color_discrete_sequence=["orange"]
        )
        fig_target = px.scatter_3d(
            self.df[self.df["body"] == "target"],
            x="x", y="y", z="z",
            size="size",
            color_discrete_sequence=["purple"]
        )
        fig_system = px.line_3d(
            self.df[self.df["body"].isin(["sun", "target"]) == False],   # noqa: E712
            x="x", y="y", z="z",
            color="body",
        )
        fig = go.Figure(data=fig_sun.data + fig_target.data + fig_system.data)
        fig.update_layout(
            scene = {
                'zaxis': {
                    'nticks': 4,
                    'range': zrange
                }
            }
        )
        fig.show()
        return fig


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


def main():
    from .states import get_body_state
    from datetime import date
    from astropy.time import Time

    now = date.today()
    time = Time(now.strftime(r'%Y-%m-%d %H:%M'), scale="utc")

    system = SunSystem(["earth", "mars"])

    walker_position, walker_velocity = get_body_state("mars", time).values()
    walker = Walker(walker_position, walker_velocity, mass=1., name="mars")

    target = np.array([0, 0, 1])

    plotter = Plotter(system=system, walker=walker, target=target)
    plotter.draw()


if __name__ == "__main__":
    main()
