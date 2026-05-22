import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jax.scipy.spatial.transform as transform
from jax import lax


from flightning.utils.pytrees import field_jnp, CustomPyTree
from flightning.utils.math import (
    quat_mul,
    quat_error_shaped,
    quat_inv,
    quat_to_rot,
    Jr_inv,
)


@jdc.pytree_dataclass
class HopfControllerParams(CustomPyTree):
    K_r: jax.Array = field_jnp(jnp.zeros(3))
    K_v: jax.Array = field_jnp(jnp.zeros(3))
    K_R: jax.Array = field_jnp(jnp.zeros(3))
    K_o: jax.Array = field_jnp(jnp.zeros(3))
    dt_policy: jax.Array = field_jnp(1.0)
    dt_position: jax.Array = field_jnp(0.005)
    chart_threshold: jax.Array = field_jnp(0.05)
    eps: jax.Array = field_jnp(1.0e-6)


@jdc.pytree_dataclass
class HopfControllerState(CustomPyTree):
    ticks: jax.Array = field_jnp(jnp.int32(0))
    eta: jax.Array = field_jnp(jnp.int32(1))
    chart: jax.Array = field_jnp(jnp.int32(1))
    yaw_offset: jax.Array = field_jnp(0.0)
    dyaw_offset: jax.Array = field_jnp(0.0)
    fc_d: jax.Array = field_jnp(0.0)
    quat_d: jax.Array = field_jnp(jnp.array([1.0, 0.0, 0.0, 0.0]))
    omega_d: jax.Array = field_jnp(jnp.zeros(3))
    motor_omega_d_prev: jax.Array = field_jnp(jnp.zeros(4))
    T_prev: jax.Array = field_jnp(jnp.zeros(4))


class HopfController:
    def __init__(
        self,
        params: HopfControllerParams,
        quadrotor,
    ):
        self.params = params
        self.quadrotor = quadrotor

        # timing
        self.N_ticks = jnp.int32(
            jnp.round(self.params.dt_position / self.quadrotor._dt_low_level)
        )

    def create_control_state(self, **kwargs):
        return HopfControllerState(**kwargs)

    def apply_controller(
        self,
        quadrotor_state,
        control_state: HopfControllerState,
        reference_quadrotor_state,
        reference_yaw: jax.Array,
        reference_dyaw: jax.Array,
        reference_eta: jax.Array,
        action: jax.Array,
    ):
        return self.__call__(
            quadrotor_state=quadrotor_state,
            reference_quadrotor_state=reference_quadrotor_state,
            control_state=control_state,
            reference_yaw=reference_yaw,
            reference_dyaw=reference_dyaw,
            reference_eta=reference_eta,
            action=action,
        )

    def __call__(
        self,
        quadrotor_state,
        control_state: HopfControllerState,
        reference_quadrotor_state,
        reference_yaw: jax.Array,
        reference_dyaw: jax.Array,
        reference_eta: jax.Array,
        action: jax.Array,
    ):
        # unpack quadrotor state
        R = quadrotor_state.R
        omega = quadrotor_state.omega
        quat = transform.Rotation.from_matrix(R).as_quat()[..., [3, 0, 1, 2]]

        def update_position_controller():
            # desired acceleration, jerk, snap
            a_des = (
                -self.params.K_r
                * (quadrotor_state.p - reference_quadrotor_state.p - action[0:3])
                - self.params.K_v * (quadrotor_state.v - reference_quadrotor_state.v)
                - self.quadrotor._g
            )
            da_des = -self.params.K_r * (
                quadrotor_state.v - reference_quadrotor_state.v
            ) - self.params.K_v * (quadrotor_state.acc - reference_quadrotor_state.acc)
            a_des_norm = jnp.maximum(jnp.linalg.norm(a_des), self.params.eps)

            eta_new = jnp.int32((reference_eta) * jnp.sign(action[3]))  # learned eta

            # compute s and ds
            s = eta_new * a_des / a_des_norm
            P = jnp.eye(3) - jnp.outer(s, s)
            ds = eta_new * (1.0 / a_des_norm) * (P @ da_des)

            # desired collective thrust
            fc_d = self.quadrotor._mass * (R.T @ a_des)[2]

            # apply chart logic
            chart_new, yaw_offset_new, dyaw_offset_new = self.update_chart(
                chart=control_state.chart,
                yaw_offset=control_state.yaw_offset,
                dyaw_offset=control_state.dyaw_offset,
                eta=eta_new,
                s=s,
                ds=ds,
            )

            # handle maintaining yaw
            yaw_d_new = eta_new * reference_yaw 
            dyaw_d = eta_new * reference_dyaw 

            yaw = yaw_d_new + control_state.yaw_offset
            dyaw = dyaw_d + control_state.dyaw_offset

            quat_d, omega_d = self.get_desired_orientation(
                chart=chart_new,
                s=s,
                ds=ds,
                yaw=yaw,
                dyaw=dyaw,
            )

            return (
                eta_new,
                chart_new,
                yaw_offset_new,
                dyaw_offset_new,
                fc_d,
                quat_d,
                omega_d,
            )

        def hold_position_controller():
            return (
                control_state.eta,
                control_state.chart,
                control_state.yaw_offset,
                control_state.dyaw_offset,
                control_state.fc_d,
                control_state.quat_d,
                control_state.omega_d,
            )

        # --- position controller ---
        run_position_controller = control_state.ticks == 0
        ticks = (control_state.ticks + 1) % self.N_ticks
        eta, chart, yaw_offset, dyaw_offset, fc_d, quat_d, omega_d = lax.cond(
            run_position_controller,
            update_position_controller,
            hold_position_controller,
        )

        # --- attitude controller ---
        e_R, quat_e = quat_error_shaped(qd=quat_d, q=quat)
        R_qe = quat_to_rot(quat_e)
        e_o = omega - R_qe.T @ omega_d
        I = self.quadrotor.inertial_matrix
        tau_d = I @ (
            - Jr_inv(e_R) @ jnp.diag(self.params.K_R) @ e_R 
            - jnp.diag(self.params.K_o) @ e_o
        )

        # --- control allocation ---
        u_des = jnp.concatenate([fc_d[None], tau_d])
        M = self.quadrotor.mixer_matrix(control_state.motor_omega_d_prev)
        motor_omega_d, T_cmd = self.quadrotor._actuator_model.allocate_control(
            u_des=u_des,
            M=M,
            T_prev=control_state.T_prev,
            motor_omega=quadrotor_state.motor_omega,
        )

        control_state_new = control_state.replace(
            ticks=ticks,
            eta=eta,
            chart=chart,
            yaw_offset=yaw_offset,
            dyaw_offset=dyaw_offset,
            fc_d=fc_d,
            quat_d=quat_d,
            omega_d=omega_d,
            motor_omega_d_prev=motor_omega_d,
            T_prev=T_cmd,
        )

        return control_state_new, motor_omega_d, u_des

    def get_desired_orientation(self, chart, s, ds, yaw, dyaw):
        a, b, c = s
        da, db, dc = ds
        eps = self.params.eps

        def calc_k_v():
            k = jnp.sqrt(2.0 * jnp.maximum(1.0 + c, eps))
            dk = dc / jnp.sqrt(2.0 * (1.0 + c))
            v = jnp.array([1.0 + c, -b, a, 0.0])
            dv = jnp.array([dc, -db, da, 0.0])
            return k, dk, v, dv

        def calc_k_v_bar():
            k = jnp.sqrt(2.0 * jnp.maximum(1.0 - c, eps))
            dk = -dc / jnp.sqrt(2.0 * (1.0 - c))
            v = jnp.array([-b, 1.0 - c, 0.0, a])
            dv = jnp.array([-db, -dc, 0.0, da])
            return k, dk, v, dv

        def calc_q_abc(k, dk, v, dv):
            q_abc = v / k
            dq_abc = dv / k - v * dk / k**2
            return q_abc, dq_abc

        def calc_q_psi(yaw, dyaw):
            q_psi = jnp.array([jnp.cos(yaw / 2.0), 0.0, 0.0, jnp.sin(yaw / 2.0)])
            dq_psi = jnp.array(
                [
                    -0.5 * dyaw * jnp.sin(yaw / 2.0),
                    0.0,
                    0.0,
                    0.5 * dyaw * jnp.cos(yaw / 2.0),
                ]
            )
            return q_psi, dq_psi

        k, dk, v, dv = lax.cond(chart == 1, calc_k_v, calc_k_v_bar)
        q_abc, dq_abc = calc_q_abc(k, dk, v, dv)
        q_psi, dq_psi = calc_q_psi(yaw, dyaw)

        q_des = quat_mul(q_abc, q_psi)
        dq_des = quat_mul(dq_abc, q_psi) + quat_mul(q_abc, dq_psi)

        # correct quaternion sign consistently
        q_des, dq_des = lax.cond(
            q_des[0] > 0.0,
            lambda _: (q_des, dq_des),
            lambda _: (-q_des, -dq_des),
            operand=None,
        )

        q_des = q_des / jnp.linalg.norm(q_des)
        omega_des = (2.0 * quat_mul(quat_inv(q_des), dq_des))[1:]
        return q_des, omega_des

    def update_chart(self, chart, yaw_offset, dyaw_offset, eta, s, ds):
        a, b, c = s
        da, db, _ = ds
        D = a * a + b * b
        D_safe = D + self.params.eps
        N = b * da - a * db

        # chart switching logic 
        # (N --> S)
        chart, yaw_offset, dyaw_offset = lax.cond(
            (eta == 1) & (chart == 1) & (c < -self.params.chart_threshold),
            lambda _: (jnp.int32(2), 2.0 * jnp.arctan2(a, b), 2.0 * N / D_safe),
            lambda _: (chart, yaw_offset, dyaw_offset),
            operand=None,
        )

        # (-eta --> +eta)
        chart, yaw_offset, dyaw_offset = lax.cond(
            (eta == 1) & (chart == 2) & (c > self.params.chart_threshold),
            lambda _: (jnp.int32(1), 0.0, 0.0),
            lambda _: (chart, yaw_offset, dyaw_offset),
            operand=None,
        )

        # (+eta --> -eta)
        chart, yaw_offset, dyaw_offset = lax.cond(
            (eta == -1) & (chart == 1) & (c < -self.params.chart_threshold),
            lambda _: (jnp.int32(2), 0.0, 0.0),
            lambda _: (chart, yaw_offset, dyaw_offset),
            operand=None,
        )

        # (S --> N)
        chart, yaw_offset, dyaw_offset = lax.cond(
            (eta == -1) & (chart == 2) & (c > self.params.chart_threshold),
            lambda _: (
                jnp.int32(1),
                -2.0 * jnp.arctan2(a, b),
                -2.0 * N / D_safe,
            ),
            lambda _: (chart, yaw_offset, dyaw_offset),
            operand=None,
        )
        return chart, yaw_offset, dyaw_offset
