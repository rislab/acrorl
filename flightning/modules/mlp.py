from typing import NamedTuple, Union, Callable
import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn


class LearnedVector(nn.Module):
    """
    Learnable fixed vector (e.g., controller gains) stored as Flax params.

    - dim: number of parameters (12)
    - init_scale: stddev for normal init (if you want random)
    - init_value: optional constant init (scalar or array-like length dim)
    - as_row: if True, returns shape (1, dim), else (dim,)
    """

    dim: int = 12
    init_scale: float = 0.0
    init_value: Union[float, jnp.ndarray, None] = 0.0
    as_row: bool = True

    @nn.compact
    def __call__(self, x=None):
        # Choose initializer
        if self.init_value is None:
            init_fn = nn.initializers.normal(stddev=self.init_scale)
        else:
            init_val = jnp.asarray(self.init_value)
            if init_val.ndim == 0:
                init_val = jnp.ones((self.dim,)) * init_val
            else:
                init_val = init_val.reshape((self.dim,))
            init_fn = lambda key, shape, dtype=jnp.float32: init_val.astype(dtype)

        theta = self.param("theta", init_fn, (self.dim,), jnp.float32)

        if self.as_row:
            theta = theta[None, :]  # (1, dim)

        # If you pass a batch of x, broadcast to (batch, dim)
        if x is not None:
            # If x is (batch, ...) -> return (batch, dim)
            if x.ndim >= 1:
                batch = x.shape[0]
                theta = jnp.broadcast_to(theta, (batch, theta.shape[-1]))
        return theta

    def initialize(self, key):
        # dummy input not needed, but keep interface similar to your MLP
        return self.init(key, None)


class MLP(nn.Module):
    """
    Simple feedforward MLP
    """

    feature_list: list
    nonlinearity: Callable = nn.relu
    initial_scale: float = 1.0
    action_bias: Union[float, jnp.ndarray] = 0.0

    @nn.compact
    def __call__(self, x):
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)

        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        return x + self.action_bias

    def initialize(self, key):
        """
        Initialize the model with random weights. Shorthand for `init`.
        :param key: random key
        :return: initial parameters
        """
        x_rand = jax.random.normal(key, (self.feature_list[0],))
        return self.init(key, x_rand)


class OrthogonalMLP(MLP):
    @nn.compact
    def __call__(self, x):
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.orthogonal(scale=self.initial_scale),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)

        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.orthogonal(scale=self.initial_scale),
            bias_init=nn.initializers.zeros,
        )(x)
        return x + self.action_bias


class PiValue(NamedTuple):
    pi: distrax.Distribution
    value: jnp.ndarray


class ActorCriticPPO(MLP):
    initial_log_std: float = 0.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        # actor
        x = obs
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        action_mean = x + self.action_bias
        # action_mean = nn.tanh(action_mean)

        action_logtstd = self.param(
            "log_std",
            nn.initializers.constant(self.initial_log_std),
            (self.feature_list[-1],),
        )
        action_std = jnp.maximum(jnp.exp(action_logtstd), 0.05)
        # create distribution object
        pi = distrax.MultivariateNormalDiag(action_mean, action_std)

        # critic
        x = obs
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            1,
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        value = jnp.squeeze(x, axis=-1)

        return PiValue(pi, value)


class Actor(MLP):
    initial_log_std: float = 0.0

    @nn.compact
    def __call__(self, obs):
        x = obs
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        action_mean = x + self.action_bias
        # action_mean = nn.tanh(action_mean)

        action_logtstd = self.param(
            "log_std",
            nn.initializers.constant(self.initial_log_std),
            (self.feature_list[-1],),
        )
        action_std = jnp.maximum(jnp.exp(action_logtstd), 0.05)
        # create distribution object
        pi = distrax.MultivariateNormalDiag(action_mean, action_std)
        return pi


class Critic(MLP):
    @nn.compact
    def __call__(self, obs):
        x = obs
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            1,
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        value = jnp.squeeze(x, axis=-1)
        return value
