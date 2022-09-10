# stellarflow

## Goal

An attempt on modelling large classical gravitational systems efficiently using `tensorflow` (pretty much solved).
Going on trying to extend the gravitational system to a `tf-agents`-compatible gym.

## Why "target"-Body masses are irrelevant:

$$
    a_i = \frac{1}{m_i} \sum_{j \neq i} F_{ji} = -\frac{1}{m_i}\sum_{j\neq i} m_im_j G \frac{r_{ji}}{\vert r_{ji}\vert^3}  = -\sum_{j\neq i} m_j G \frac{r_{ji}}{\vert r_{ji}\vert^3} 
$$
