import jax
from jax import numpy as jnp
import jax_dataclasses as jdc
from .base_trajectory import BaseTrajectory
from flightning.objects import QuadrotorState


@jdc.pytree_dataclass
class ConstantTrajectory(BaseTrajectory):
    init_quadrotor_state: QuadrotorState
    init_yaw: jax.Array
    eta: jax.Array
    duration: float

    target_quadrotor_state: QuadrotorState

    def __call__(self, time):
        target_quadrotor_state = self.target_quadrotor_state
        target_yaw = self.init_yaw
        target_dyaw = jnp.zeros_like(self.init_yaw)
        eta = self.eta
        return target_quadrotor_state, target_yaw, target_dyaw, eta
