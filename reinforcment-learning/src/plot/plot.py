import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from seaborn import scatterplot
import matplotlib.pyplot as plt


from .plot_data_generators import (
    generate_plot_data_system,
    generate_plot_data_target,
    generate_plot_data_walker,
    generate_plot_data_env
)
from .plot_data_generators import _body_colname, _position_colnames
x_col, y_col, z_col = _position_colnames


class Plotter:

    n_bodies = 0

    def __init__(self, object_dict: dict):
        """Provide a dictionary containing one or more of:
        ```
        {
            'env': WalkerSystemEnv,
            'system': SunSystem,
            'walker': Walker,
            'target': Target
        }
        ```
        """

        if "walker" in object_dict:
            walker = object_dict.get("walker")
            df_walker = generate_plot_data_walker(walker)
            self.df = df_walker
            self.n_bodies += 1

        if "system" in object_dict:
            system = object_dict.get("system")
            df_system = generate_plot_data_system(system)
            self.n_bodies += system.n_bodies
            self.df = pd.concat([self.df, df_system], axis=0)

        if "target" in object_dict:
            target = object_dict.get("target")
            df_target = generate_plot_data_target(target)
            self.n_bodies += system.n_bodies
            self.df = pd.concat([self.df, df_target], axis=0)

        if "env" in object_dict:
            env = object_dict.get("env")
            self.df = generate_plot_data_env(env)
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

        df_sun = self.df[self.df[_body_colname] == "sun"]
        fig_sun = px.scatter_3d(
            df_sun,
            x=x_col, y=y_col, z=z_col, color=_body_colname,
            size="size",
            color_discrete_sequence=["orange"]
        )

        df_target = self.df[self.df[_body_colname] == "target"]
        fig_target = px.scatter_3d(
            df_target,
            x=x_col, y=y_col, z=z_col,
            size="size",
            color_discrete_sequence=["purple"]
        )

        df_system = self.df[ ~ self.df[_body_colname].isin(["sun", "target"])]
        fig_system = px.line_3d(
            df_system,
            x=x_col, y=y_col, z=z_col,
            color=_body_colname,
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
        fig = scatterplot(self.df, x=x_col, y=y_col, hue=_body_colname, size="size")
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
    from ..data import get_body_state
    from ..components import SunSystem, Walker

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
