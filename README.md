# stellarflow

### Goal

An attempt on modelling large classical gravitational systems efficiently using `tensorflow` (pretty much solved). Going on trying to extend the gravitational system to a `tf-agents`-compatible gym.

## Compatibility Issue with `numpy.bool`

The `numpy.bool` dtype is deprecated. Yet standard installations of TensorFlow (v2.10) and tf-Agents (v0.14) for Windows (GPU) using a conda-pip mix (sorry, I know, I should switch) yields to a compatibility issue because there is a leftover `np.bool` in the typing submodule of tf-Agents. As far as I have tested it, it is save to simply remove the `np.bool` entry or replace it with standard python `bool`.

### Why "target"-Body masses are irrelevant:

$$
    a_i = \frac{1}{m_i} \sum_{j \neq i} F_{ji} = -\frac{1}{m_i}\sum_{j\neq i} m_im_j G \frac{r_{ji}}{\vert r_{ji}\vert^3}  = -\sum_{j\neq i} m_j G \frac{r_{ji}}{\vert r_{ji}\vert^3} 
$$
