## Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import matplotlib.pyplot as plt




## Settings
tf.random.set_seed(42)
np.random.seed(42)
AU = 147095000000.





@tf.function
def pairwise_distances(X: tf.Tensor) -> tf.Tensor:
    # r = tf.reduce_sum(X * X, 1)
    # r = tf.reshape(r, [-1, 1])
    # D = r - 2.*tf.matmul(X, tf.transpose(X)) + tf.transpose(r)
    return tf.expand_dims(X, axis=0) - tf.expand_dims(X, axis=1)

@tf.function
def acceleration(X: tf.Tensor, M: tf.Tensor, G: float = 6.67430e-11 / 149597870700.**3. * 86400.**2. * 1.98847e30) -> tf.Tensor:

    ## Calculating Pairwise Distances
    D = pairwise_distances(X)

    ## Masking and reshaping X and M to drop self-differences:
    mask_reps = tf.cast(D.shape[0], tf.int16)
    D = tf.reshape(D, (-1, 3))
    mask = tf.tile([False] + (mask_reps)*[True], [mask_reps])[:D.shape[0]] ## Would be optimal to store as constant in a class.
    D = tf.boolean_mask(D, mask, axis=0)

    D = tf.reshape(D, (mask_reps, (mask_reps-1), -1))
    M = tf.tile(M, [mask_reps])
    M = tf.reshape(M, (-1, 1))
    M = tf.boolean_mask(M, mask, axis=0)
    M = tf.reshape(M, (mask_reps, -1, 1))

    ## Calculating |xi-xj|^(-3)
    d_inv_cube = tf.pow(tf.reduce_sum(D*D, axis=-1, keepdims=True) + 1e-10, -3./2.) # 1e-10 to smooth numerical overflow
    
    ## Calculating pairwise Force
    F = G *  M * D / d_inv_cube

    ## Summation over each other body
    F = tf.reduce_sum(F, axis=1)

    ## Combining to R6 acceleration
    dQ = tf.concat([Q[:, 3:], F], axis=-1)
    
    return dQ

def return_ode_fn(M):
    @tf.function
    def ode_fn(Q):
        return acceleration(Q, M)
    return ode_fn








Q0 = tf.Variable([
    [0., 0., 0., 0., 0., 0.], 
    [1., 0., 0., 0.,    30300./AU, 0.],
    [2., 0., 0., 0., 2.*30300./AU, 0.],
    [3., 0., 0., 0., 3.*30300./AU, 0.]
])
M0 = tf.constant([
    1., 
    3.0025e-6,
    3.0025e-6, 
    3.0025e-6
])


acceleration(Q0[:, :3], M0)