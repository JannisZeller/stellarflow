
import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

from src.system import NBodySystem


class SystemPlotter:
    def __init__(self, system: NBodySystem, n_sample: int=None):
        self.system = system
        self.sample = self._get_body_sample(n_sample)

    def _get_body_sample(self, n_sample) -> np.ndarray | list:
        if n_sample is not None:
            if self.system.body_count > n_sample:
                sample = np.random.choice(
                    np.arange(self.system.body_count),
                    size=n_sample,
                    replace=False,
                )
        else:
            sample = range(self.system.body_count)
        return sample

    def show(self, mode: str="3d", figsize: tuple=(5, 5), **kwargs):
        if mode not in ["3d", "2d"]:
            raise ValueError('`mode` must be of ["3d", "2d"].')

        if mode == "3d":
            if 'zsize' in kwargs.keys():
                warn("The `zsize` kwarg is only used in `2d` mode.")
            self._plot_system_history_3d(figsize=figsize)

        if mode == "2d":
            self._plot_system_history_2d(
                figsize=figsize, zsize=kwargs.get("zsize", True)
            )

    def _plot_system_history_3d(self, figsize: tuple=(5, 5)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        for idx in self.sample:
            ax.plot(
                self.system.state_history[:, idx, 0],
                self.system.state_history[:, idx, 1],
                self.system.state_history[:, idx, 2],
            )
        plt.show()

    def _plot_system_history_2d(self, figsize: tuple=(5, 5), zsize: bool=True):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for idx in self.sample:
            if zsize:
                ax.scatter(
                    self.system.state_history[:, idx, 0],
                    self.system.state_history[:, idx, 1],
                    s=0.75
                    * np.clip(
                        self.system.state_history[:, idx, 2].numpy(),
                        a_min=1e-10,
                        a_max=np.inf,
                    ),
                )
            else:
                ax.plot(
                    self.system.state_history[:, idx, 0],
                    self.system.state_history[:, idx, 1],
                )
        plt.show()
