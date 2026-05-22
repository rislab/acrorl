import jax
from jax import numpy as jnp
import jax_dataclasses as jdc
from jax.scipy.linalg import expm
from .base_trajectory import BaseTrajectory
from flightning.objects import QuadrotorState
from flightning.utils.math import skew
from flightning.utils.trajectory import minimum_snap


@jdc.pytree_dataclass
class LoopTrajectory(BaseTrajectory):
    init_quadrotor_state: QuadrotorState
    init_yaw: jax.Array
    eta: jax.Array
    duration: float

    angular_speed: float
    radius: float
    tilt_angle: float
    dynamic_tilt: jdc.Static[bool]

    def __call__(self, time):
        cosine = jnp.cos(self.angular_speed * time)
        sine = jnp.sin(self.angular_speed * time)

        p_base = jnp.array(
            [self.radius * cosine - self.radius, self.radius * sine, 0.0]
        )
        v_base = jnp.array(
            [
                -self.radius * self.angular_speed * sine,
                self.radius * self.angular_speed * cosine,
                0.0,
            ]
        )
        acc_base = jnp.array(
            [
                -self.radius * self.angular_speed**2 * cosine,
                -self.radius * self.angular_speed**2 * sine,
                0.0,
            ]
        )

        if self.dynamic_tilt:
            alpha, dalpha, ddalpha, _ = minimum_snap(
                time, self.duration, self.tilt_angle
            )
        else:
            alpha, dalpha, ddalpha = self.tilt_angle, 0.0, 0.0

        v = jnp.array([1.0, 0.0, 0.0])
        zeta = alpha * v
        dzeta = dalpha * v
        ddzeta = ddalpha * v
        zeta_hat = skew(zeta)
        dzeta_hat = skew(dzeta)
        ddzeta_hat = skew(ddzeta)

        R = expm(zeta_hat)
        p = R @ p_base + self.init_quadrotor_state.p
        v = R @ (v_base + dzeta_hat @ p_base)
        acc = R @ (
            acc_base
            + 2.0 * dzeta_hat @ v_base
            + (ddzeta_hat + dzeta_hat @ dzeta_hat) @ p_base
        )

        reference_quadrotor_state = QuadrotorState(p=p, v=v, acc=acc)
        reference_yaw = self.init_yaw
        reference_dyaw = jnp.zeros_like(self.init_yaw)
        eta = self.eta

        return reference_quadrotor_state, reference_yaw, reference_dyaw, eta
