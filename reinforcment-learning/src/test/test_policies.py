import tensorflow as tf


def one_dim_test_policy(observations):
    """Executes a policy which always returns a boost towards the target in
    1 dimension along the z-axis.
    Works only in `env.OneDimDiscreteAction`-based environments with
    `batch_size == 1`.
    """
    obs = observations.observation

    walker_position = obs['walker-state'][..., :3]

    if 'target' in obs:
        target_position = obs['target']
        diff_to_target = target_position - walker_position
    if 'diff-to-target' in obs:
        diff_to_target = obs['diff-to-target']


    z_diff = diff_to_target[-1, -1]

    if tf.abs(z_diff) < 1e-3:
        action = 2

    if z_diff < 0:
        action =  1

    if z_diff > 0:
        action =  0

    action = tf.reshape(action, [-1])

    return action
