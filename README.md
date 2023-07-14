# stellarflow

### Contents

In `./n-body-simulation` I do some fun simulations of (potentially pretty large) gravitational systems with classical non-relativistc mechanics. I use TensorFlow to accelerate the simulation.

In `./reinforcment-learning` I am currently working on training a little reinforcement learning agent that can perform some maneuvers flying around in graviational systems. I am using the TensorFlow-Agents library there. Previously I built the "gym"-environment based on my works from `./n-body-simulation`, but then I decided to separate the two things. Building the RL-approach on top of the simulation would yield unnecessairy loads for computing the trajectories, which wouldn't even be realistic. I decided to rely on known data here and focus only on the RL stuff. Learning an agent, that can perform full maneuvers etc. seems to be very costly and time intensive (Sullivan & Bosanac, [2020](https://www.colorado.edu/faculty/bosanac/sites/default/files/attached-files/2020_sulbos_aiaa.pdf); Kolosa, [2019](https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4537&context=dissertations)). The presented approach therefore primarily presents a working concept for an environment implementation in TF-Agents.

## Compatibility Issue with `numpy.bool`

The `numpy.bool` dtype is deprecated. Yet standard installations of TensorFlow (v2.10) and tf-Agents (v0.14) for Windows (GPU) using a conda-pip mix (sorry, I know, I should switch) yields to a compatibility issue because there is a leftover `np.bool` in the typing submodule of tf-Agents. As far as I have tested it, it is save to simply remove the `np.bool` entry or replace it with standard python `bool`.

### Why "target"-Body masses are irrelevant:

$$
    a_i = \frac{1}{m_i} \sum_{j \neq i} F_{ji} = -\frac{1}{m_i}\sum_{j\neq i} m_im_j G \frac{r_{ji}}{\vert r_{ji}\vert^3}  = -\sum_{j\neq i} m_j G \frac{r_{ji}}{\vert r_{ji}\vert^3}
$$
