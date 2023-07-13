
from tf_agents.environments import (
    tf_py_environment,
    tf_environment,
    validate_py_environment
)

from .env import WalkerSystemEnv


def tfpy_env_wrapper(
        environment: WalkerSystemEnv,
        validate: bool=False
    ) -> tf_environment:
    if validate:
        validate_py_environment(environment)
    tf_env = tf_py_environment.TFPyEnvironment(environment)
    tf_env.system = environment.system
    tf_env.walker = environment.walker
    tf_env.target = environment.target
    return tf_env
