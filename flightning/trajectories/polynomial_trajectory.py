import jax
from jax import numpy as jnp
import jax_dataclasses as jdc

from .base_trajectory import BaseTrajectory
from flightning.objects import QuadrotorState
from flightning.utils.trajectory import (
    generate_3d_trajectory,
    evaluate_polynomial_reference,
)


@jax.tree_util.register_pytree_node_class
class PolynomialTrajectory(BaseTrajectory):
    def __init__(
        self,
        init_quadrotor_state,
        init_yaw,
        eta,
        duration,
        target_quadrotor_state,
        apogee=0.0,
        segment_times=(),
        eta_sequence=(),
        free_fall_idx=(),
    ):
        self.init_quadrotor_state = init_quadrotor_state
        self.init_yaw = init_yaw
        self.eta = eta
        self.duration = float(duration)
        self.target_quadrotor_state = target_quadrotor_state
        self.apogee = float(apogee)
        self.segment_times = tuple(segment_times)
        self.eta_sequence = tuple(eta_sequence)
        self.free_fall_idx = tuple(free_fall_idx)

        self._build_coefficients()

    def _build_coefficients(self):
        p0 = self.init_quadrotor_state.p
        pf = self.target_quadrotor_state.p

        if len(self.segment_times) == 0:
            times = jnp.array([0.0, self.duration])
        else:
            times = jnp.asarray(self.segment_times)

        # Case A: flip-style polynomial with midpoint apex
        if times.shape[0] == 3:
            midpoint = jnp.array(
                [
                    0.5 * (p0[0] + pf[0]),
                    0.5 * (p0[1] + pf[1]),
                    p0[2] + self.apogee,
                ]
            )

            waypoints = jnp.array(
                [
                    p0,
                    midpoint,
                    pf,
                ]
            )

        # Case B: normal polynomial from initial to target
        elif times.shape[0] == 2:
            waypoints = jnp.array(
                [
                    p0,
                    pf,
                ]
            )

        else:
            raise ValueError(
                "PolynomialTrajectory currently supports either 2 or 3 waypoint times."
            )

        yaw_waypoints = jnp.ones(times.shape[0]) * self.init_yaw
        n_segments = times.shape[0] - 1

        if len(self.eta_sequence) == 0:
            eta_sequence = jnp.ones((n_segments,), dtype=jnp.int32) * self.init_eta
        else:
            eta_sequence = tuple(self.eta_sequence)

        if len(eta_sequence) != n_segments:
            raise ValueError(
                f"eta_sequence must have length {n_segments}, got {len(eta_sequence)}"
            )

        free_fall_idx = tuple(self.free_fall_idx)
        cx, cy, cz, cyaw, eta_sequence_out = generate_3d_trajectory(
            times=times,
            waypoints=waypoints,
            yaw_waypoints=yaw_waypoints,
            s=eta_sequence,
            start_v=self.init_quadrotor_state.v,
            free_fall_idx=free_fall_idx,
        )

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "cx", cx)
        object.__setattr__(self, "cy", cy)
        object.__setattr__(self, "cz", cz)
        object.__setattr__(self, "cyaw", cyaw)
        object.__setattr__(
            self,
            "eta_sequence_arr",
            jnp.asarray(eta_sequence_out, dtype=jnp.int32),
        )

    def __call__(self, time):
        t_clipped = jnp.clip(time, 0.0, self.duration)

        p, v, acc, jrk, snp, yaw, dyaw, ddyaw, eta = evaluate_polynomial_reference(
            self.times,
            self.cx,
            self.cy,
            self.cz,
            self.cyaw,
            self.eta_sequence_arr,
            t_clipped,
        )

        # Hold final derivatives at zero after the segment is done
        is_moving = time < self.duration
        v = jnp.where(is_moving, v, 0.0)
        acc = jnp.where(is_moving, acc, 0.0)
        jrk = jnp.where(is_moving, jrk, 0.0)

        reference_quadrotor_state = self.init_quadrotor_state.replace(
            p=p,
            v=v,
            acc=acc,
            jrk=jrk,
        )

        return reference_quadrotor_state, yaw, dyaw, eta

    def tree_flatten(self):
        children = (
            self.init_quadrotor_state,
            self.target_quadrotor_state,
            self.init_yaw,
            self.eta,
            self.times,
            self.cx,
            self.cy,
            self.cz,
            self.cyaw,
            self.eta_sequence_arr,
        )
        aux = (
            self.duration,
            self.apogee,
            self.segment_times,
            self.eta_sequence,
            self.free_fall_idx,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls.__new__(cls)

        (
            init_quadrotor_state,
            target_quadrotor_state,
            init_yaw,
            eta,
            times,
            cx,
            cy,
            cz,
            cyaw,
            eta_sequence_arr,
        ) = children

        (
            duration,
            apogee,
            segment_times,
            eta_sequence,
            free_fall_idx,
        ) = aux

        obj.init_quadrotor_state = init_quadrotor_state
        obj.target_quadrotor_state = target_quadrotor_state
        obj.init_yaw = init_yaw
        obj.eta = eta
        obj.duration = duration
        obj.apogee = apogee
        obj.segment_times = segment_times
        obj.eta_sequence = eta_sequence
        obj.free_fall_idx = free_fall_idx

        obj.times = times
        obj.cx = cx
        obj.cy = cy
        obj.cz = cz
        obj.cyaw = cyaw
        obj.eta_sequence_arr = eta_sequence_arr

        return obj
