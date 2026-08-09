import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from flightning.utils.pytrees import field_jnp, CustomPyTree


@jdc.pytree_dataclass
class CtbrControllerParams(CustomPyTree):
    K_o: jax.Array = field_jnp(jnp.array([10.0, 10.0, 21.0]))


@jdc.pytree_dataclass
class CtbrControllerState(CustomPyTree):
    motor_omega_d_prev: jax.Array = field_jnp(jnp.zeros(4))
    T_prev: jax.Array = field_jnp(jnp.zeros(4))


class CtbrController:
    """
    Collective-thrust body-rate (CTBR) controller.
    """

    def __init__(self, params: CtbrControllerParams, quadrotor):
        self.params = params
        self.quadrotor = quadrotor

    def create_control_state(self, **kwargs):
        return CtbrControllerState(**kwargs)

    def apply_controller(
        self,
        quadrotor_state,
        control_state: CtbrControllerState,
        reference_quadrotor_state,
        reference_yaw: jax.Array,
        reference_dyaw: jax.Array,
        reference_eta: jax.Array,
        action: jax.Array,
    ):
        # reference_quadrotor_state --> ignored
        # reference_yaw --> ignored
        # reference_dyaw --> ignored
        return self.__call__(
            quadrotor_state=quadrotor_state,
            control_state=control_state,
            action=action,
        )

    def __call__(
        self,
        quadrotor_state,
        control_state: CtbrControllerState,
        action: jax.Array,
    ):
        """
        action = [T_cmd, omega_x_cmd, omega_y_cmd, omega_z_cmd]
          T_cmd:     collective thrust command [N]
          omega_cmd: desired body rates [rad/s]
        """
        T_cmd = action[0]
        omega_cmd = action[1:4]
        omega = quadrotor_state.omega

        # body-rate P control + Coriolis feedforward
        I = self.quadrotor.inertial_matrix
        omega_err = omega_cmd - omega
        tau_d = I @ (jnp.diag(self.params.K_o) @ omega_err) + jnp.cross(
            omega, I @ omega
        )

        # Optimal control allocation
        u_des = jnp.concatenate([T_cmd[None], tau_d])
        M = self.quadrotor.mixer_matrix(control_state.motor_omega_d_prev)
        motor_omega_d, T_cmd_out = self.quadrotor._actuator_model.allocate_control(
            u_des=u_des,
            M=M,
            T_prev=control_state.T_prev,
            motor_omega=quadrotor_state.motor_omega,
        )

        control_state_new = control_state.replace(
            motor_omega_d_prev=motor_omega_d,
            T_prev=T_cmd_out,
        )

        return control_state_new, motor_omega_d, u_des