import os
import jax
import yaml
import chex
import numpy as np
import jax_dataclasses as jdc
import jax.scipy.spatial.transform as transform
from jax import numpy as jnp, lax

from flightning import FLIGHTNING_PATH
from flightning.utils.pytrees import field_jnp, CustomPyTree
from flightning.controllers import (
    HopfControllerParams,
    HopfController,
    HopfControllerState,
)
from flightning.utils.math import rotation_matrix_from_vector, proj_gravity, sigmoid
from flightning.simulation import (
    ActuatorModelParams,
    ActuatorModel,
    BodyDragParams,
    compute_drag_force,
    VelocityKickParams,
    simulate_velocity_kick,
)


@jdc.pytree_dataclass
class QuadrotorState(CustomPyTree):
    p: jax.Array = field_jnp([0.0, 0.0, 0.0])
    R: jax.Array = field_jnp(jnp.eye(3))
    v: jax.Array = field_jnp([0.0, 0.0, 0.0])
    omega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    domega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    motor_omega: jax.Array = field_jnp([0.0, 0.0, 0.0, 0.0])
    acc: jax.Array = field_jnp([0.0, 0.0, 0.0])
    jrk: jax.Array = field_jnp([0.0, 0.0, 0.0])
    u: jax.Array = field_jnp([0.0, 0.0, 0.0, 0.0])
    t: jax.Array = field_jnp(0.0)
    dr_key: chex.PRNGKey = field_jnp(jax.random.key(0))

    def detached(self):
        return QuadrotorState(
            p=lax.stop_gradient(self.p),
            R=lax.stop_gradient(self.R),
            v=lax.stop_gradient(self.v),
            omega=lax.stop_gradient(self.omega),
            domega=lax.stop_gradient(self.domega),
            motor_omega=lax.stop_gradient(self.motor_omega),
            acc=lax.stop_gradient(self.acc),
            jrk=lax.stop_gradient(self.acc),
            u=lax.stop_gradient(self.u),
            t=lax.stop_gradient(self.t),
            dr_key=lax.stop_gradient(self.dr_key),
        )

    def as_vector(self):
        return jnp.concatenate(
            [
                self.p,
                self.R.flatten(),
                self.v,
                self.omega,
                self.domega,
                self.motor_omega,
            ]
        )

    @classmethod
    def from_vector(cls, vector):
        p = vector[:3]
        R = vector[3:12].reshape(3, 3)
        v = vector[12:15]
        omega = vector[15:18]
        domega = vector[18:21]
        motor_omega = vector[21:25]
        u = vector[25:29]
        return cls(p, R, v, omega, domega, motor_omega, u)


class Quadrotor:
    """full quadrotor model based on agilicious framework"""

    def __init__(
        self,
        *,
        mass=0.5,  # [kg]
        tbm_fr=jnp.array([0.09, -0.09, 0.0]),  # [m]
        tbm_bl=jnp.array([-0.09, 0.09, 0.0]),  # [m]
        tbm_br=jnp.array([-0.09, -0.09, 0.0]),  # [m]
        tbm_fl=jnp.array([0.09, 0.09, 0.0]),  # [m]
        inertia=jnp.array([0.009, 0.009, 0.012]),  # [kgm^2]
        cd_h=0.0,  # [1]
        cd_v=0.0,  # [1]
        fA_x=1.0e-3,  # [m^2]
        fA_y=1.0e-3,  # [m^2]
        fA_z=1.0e-2,  # [m^2]
        rho=1.2,  # [kg/m^3]
        cd_dr=False,
        g=jnp.array([0.0, 0.0, -9.81]),  # [m/s^2]
        motor_omega_min=-2000.0,  # [rad/s]
        motor_omega_max=2000.0,  # [rad/s]
        motor_omega_0=jnp.array([-200.0, 200.0]),  # [rad/s]
        motor_omega_dead=50.0,  # [rad/s]
        motor_xi=56.5,  # [rad/s]
        motor_tau_pos=0.033,  # [s]
        motor_tau_neg=0.033,  # [s]
        motor_tau_dr=False,  # domain randomization
        motor_inertia=5.0e-06,  # [kgm^2]
        motor_directions=jnp.array([1.0, 1.0, -1.0, -1.0]),  # [1]
        cT_pos=jnp.array([2.510e-6, -0.0016, 1.241]),  # [N/(rad/s)^2, N/(rad/s), N]
        cT_neg=jnp.array([1.648e-6, -0.0015, -1.179]),  # [N/(rad/s)^2, N/(rad/s), N]
        cT_dr=False,  # domain randomization
        full_thrust_model=False,  # full or simple thrust model
        ms_pos=0.0015,  # [Nm/N]
        ms_neg=0.0015,  # [Nm/N]
        g_bz_xi=0.05,  # [1]
        Q_u=jnp.array([1.0, 8.0, 8.0, 5.0]),  # control allocation weighting
        use_opt_alloc=False,  # optimal control allocation
        action_max=jnp.array([3.0, 3.0, 3.0]),  # [p_d, yaw_d]
        action_min=jnp.array([3.0, 3.0, 3.0]),
        ref_controller="hopf",  #  [reference controller]
        kick_frequency=5.0,  # [Hz]
        kick_magnitude=1.5,  # [m/s]
        dt_low_level=0.001,  # [s]
    ):
        self._mass = mass
        self._tbm_fr = tbm_fr
        self._tbm_bl = tbm_bl
        self._tbm_br = tbm_br
        self._tbm_fl = tbm_fl
        self._inertia = inertia
        self._cd_h = cd_h
        self._cd_v = cd_v
        self._fA_x = fA_x
        self._fA_y = fA_y
        self._fA_z = fA_z
        self._rho = rho
        self._cd_dr = cd_dr
        self._g = g
        self._motor_omega_min = motor_omega_min
        self._motor_omega_max = motor_omega_max
        self._motor_omega_0 = motor_omega_0
        self._motor_omega_dead = motor_omega_dead
        self._motor_xi = motor_xi
        self._motor_tau_pos = motor_tau_pos
        self._motor_tau_neg = motor_tau_neg
        self._motor_tau_dr = motor_tau_dr
        self._motor_inertia = motor_inertia
        self._motor_directions = motor_directions
        self._cT_pos = cT_pos
        self._cT_neg = cT_neg
        self._cT_dr = cT_dr
        self._full_thrust_model = full_thrust_model
        self._ms_pos = ms_pos
        self._ms_neg = ms_neg
        self._g_bz_xi = g_bz_xi
        self._Q_u = Q_u
        self._use_opt_alloc = use_opt_alloc
        self._action_max = action_max
        self._action_min = action_min
        self._kick_frequency = kick_frequency
        self._kick_magnitude = kick_magnitude
        self._dt_low_level = dt_low_level

        # drag model
        self._drag_params = BodyDragParams(
            horizontal_drag_coefficient=self._cd_h,
            vertical_drag_coefficient=self._cd_v,
            frontarea_x=self._fA_x,
            frontarea_y=self._fA_y,
            frontarea_z=self._fA_z,
            air_density=self._rho,
            cd_dr=self._cd_dr,
        )

        # actuator model
        self._actuator_params = ActuatorModelParams(
            motor_omega_max=self._motor_omega_max,
            motor_omega_min=self._motor_omega_min,
            motor_omega_0=self._motor_omega_0,
            motor_omega_dead=self._motor_omega_dead,
            motor_tau_pos=self._motor_tau_pos,
            motor_tau_neg=self._motor_tau_neg,
            motor_tau_dr=self._motor_tau_dr,
            motor_inertia=self._motor_inertia,
            motor_directions=self._motor_directions,
            full_thrust_model=self._full_thrust_model,
            cT_pos=self._cT_pos,
            cT_neg=self._cT_neg,
            cT_dr=self._cT_dr,
            motor_xi=self._motor_xi,
            Q_u=self._Q_u,
            use_opt_alloc=self._use_opt_alloc,
        )
        self._actuator_model = ActuatorModel(self, self._actuator_params)

        # velocity kick
        self._velocity_kick_params = VelocityKickParams(
            kick_frequency=self._kick_frequency, kick_magnitude=self._kick_magnitude
        )

        hopf_param_path = os.path.join(
            FLIGHTNING_PATH, "controllers", "config", "hopf.yaml"
        )
        with open(hopf_param_path, "r") as f:
            cfg = yaml.safe_load(f)
        controller_params = HopfControllerParams(
            K_r=jnp.array(cfg["K_r"]),
            K_v=jnp.array(cfg["K_v"]),
            K_R=jnp.array(cfg["K_R"]),
            K_o=jnp.array(cfg["K_o"]),
            dt_position=cfg["dt_position"],
            chart_threshold=cfg["chart_threshold"],
            eps=cfg["eps"],
        )
        self._reference_controller = HopfController(
            params=controller_params,
            quadrotor=self,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Quadrotor":
        with open(path) as stream:
            try:
                config = yaml.safe_load(stream)
                return cls.from_dict(config)
            except yaml.YAMLError as exc:
                raise exc

    @classmethod
    def from_dict(cls, config: dict) -> "Quadrotor":
        return cls(
            mass=config["mass"],
            tbm_fr=jnp.array(config["tbm_fr"]),
            tbm_bl=jnp.array(config["tbm_bl"]),
            tbm_br=jnp.array(config["tbm_br"]),
            tbm_fl=jnp.array(config["tbm_fl"]),
            inertia=jnp.array(config["inertia"]),
            cd_h=config["cd_h"],
            cd_v=config["cd_v"],
            fA_x=config["fA_x"],
            fA_y=config["fA_y"],
            fA_z=config["fA_z"],
            rho=config["rho"],
            g=jnp.array(config["g"]),
            motor_omega_min=config["motor_omega_min"],
            motor_omega_max=config["motor_omega_max"],
            motor_omega_0=jnp.array(config["motor_omega_0"]),
            motor_omega_dead=config["motor_omega_dead"],
            motor_xi=config["motor_xi"],
            motor_tau_pos=config["motor_tau_pos"],
            motor_tau_neg=config["motor_tau_neg"],
            motor_tau_dr=config["motor_tau_dr"],
            motor_inertia=config["motor_inertia"],
            motor_directions=jnp.array(config["motor_directions"]),
            cT_pos=jnp.array(config["cT_pos"]),
            cT_neg=jnp.array(config["cT_neg"]),
            cT_dr=config["cT_dr"],
            full_thrust_model=config["full_thrust_model"],
            ms_pos=config["ms_pos"],
            ms_neg=config["ms_neg"],
            g_bz_xi=config["g_bz_xi"],
            Q_u=jnp.array(config["Q_u"]),
            use_opt_alloc=config["use_opt_alloc"],
            action_max=jnp.array(config["action_max"]),
            action_min=jnp.array(config["action_min"]),
            ref_controller=config["ref_controller"],
            kick_frequency=config["kick_frequency"],
            kick_magnitude=config["kick_magnitude"],
            dt_low_level=config["dt_low_level"],
        )

    @property
    def hovering_motor_speed(self) -> float:
        return self._actuator_model.get_motor_omega_d(
            T_d=self._mass * jnp.abs(self._g[2]) / 4,
            motor_omega=jnp.ones(4) * self._motor_omega_max,
        )

    @property
    def inverted_hovering_motor_speed(self) -> float:
        return self._actuator_model.get_motor_omega_d(
            T_d=-self._mass * jnp.abs(self._g[2]) / 4,
            motor_omega=jnp.ones(4) * self._motor_omega_min,
        )

    @property
    def inertial_matrix(self):
        return np.diag(self._inertia)

    def default_state(self):
        """default state used for unit test"""
        rot = transform.Rotation.from_euler("zyx", jnp.array([0.0, 0.0, 0.0]))
        R = rot.as_matrix()
        hovering_motor_speeds = jnp.ones(4) * self.hovering_motor_speed
        return QuadrotorState(motor_omega=hovering_motor_speeds, R=R)

    def default_inverted_state(self):
        """default inverted state used for unit test"""
        rot = transform.Rotation.from_euler("zyx", jnp.array([0.0, jnp.pi, 0.0]))
        R = rot.as_matrix()
        hovering_motor_speeds = jnp.ones(4) * self.inverted_hovering_motor_speed
        return QuadrotorState(motor_omega=hovering_motor_speeds, R=R)

    def create_state(self, p, R, v, **kwargs):
        g_b = proj_gravity(R)
        hovering_motor_speed = jnp.ones(4) * self.hovering_speed_map(g_b[2])
        if "motor_omega" not in kwargs.keys():
            kwargs["motor_omega"] = hovering_motor_speed
        return QuadrotorState(p, R, v, **kwargs)

    def step(
        self,
        quadrotor_state: QuadrotorState,
        control_state: HopfControllerState,
        reference_quadrotor_state: QuadrotorState,
        reference_yaw: jax.Array,
        reference_dyaw: jax.Array,
        reference_eta: jax.Array,
        action: jax.Array,
        dt: jax.Array,
    ) -> QuadrotorState:
        """
        :param state: quadrotor state
        :param control_state: controller state
        :param action:
        :param dt: time step length [s]
        :return: next state of the quadrotor
        """

        # round dt to 5 decimal places to avoid numerical issues
        dt = np.round(dt, 5)
        if dt <= 0.0:
            return quadrotor_state, control_state

        def control_fn(carry, _unused):
            """
            low-level controller and dynamics.
            runs by default at 1 kHz.
            """
            quadrotor_state, control_state = carry

            # apply control (step control state)
            control_state, motor_omega_d, u = (
                self._reference_controller.apply_controller(
                    quadrotor_state=quadrotor_state,
                    control_state=control_state,
                    reference_quadrotor_state=reference_quadrotor_state,
                    reference_eta=reference_eta,
                    reference_yaw=reference_yaw,
                    reference_dyaw=reference_dyaw,
                    action=action,
                )
            )

            # apply dynamics (step quadrotor state)
            quadrotor_state = self._dynamics(
                state=quadrotor_state,
                motor_omega_d=motor_omega_d,
                u=u,
                dt=self._dt_low_level,
            )
            return (quadrotor_state, control_state), None

        N = np.ceil(dt / self._dt_low_level).item()
        # check if dt is a multiple of dt_low_level
        assert np.isclose(
            N * self._dt_low_level, dt
        ), f"dt ({dt}) must be a multiple of dt_low_level ({self._dt_low_level})"

        (quadrotor_state_new, control_state_new), _ = lax.scan(
            control_fn, (quadrotor_state, control_state), length=N
        )

        return quadrotor_state_new, control_state_new

    def _dynamics(self, state: QuadrotorState, motor_omega_d, u, dt):
        """semi-implicit euler stepping of quadrotor dynamics"""

        # unpack state
        p = state.p
        R = state.R
        v = state.v
        acc = state.acc
        omega = state.omega
        motor_omega = state.motor_omega
        t = state.t

        # domain randomization keys
        key_thrust, key_lag, key_drag, key_kick = jax.random.split(state.dr_key, 4)

        # motor thrust
        T = self._actuator_model.get_T(key_thrust, motor_omega)
        wrench = self.mixer_matrix(motor_omega) @ T

        # drag model
        f_drag = compute_drag_force(state, key_drag, self._drag_params)

        # velocity kick
        v_kick = simulate_velocity_kick(dt, key_kick, self._velocity_kick_params)

        # total force, linear accel, linear vel
        f_vec = jnp.array([0, 0, wrench[0]]) + f_drag
        acc_new = self._g + R @ f_vec / self._mass

        # semi-implicit euler update
        v_new = v + v_kick + dt * acc_new
        p_new = p + dt * v_new

        # update jerk
        jrk = (acc_new - acc) / dt

        # step actuator dynamics
        motor_inertia_torque, motor_omega_new = self._actuator_model.step(
            key_lag, motor_omega, motor_omega_d, dt
        )

        # body torques, angular accel, angular vel
        J = self.inertial_matrix
        tau = wrench[1:]
        domega_new = jnp.linalg.solve(
            J, tau - jnp.cross(omega, J @ omega) + motor_inertia_torque
        )
        omega_new = omega + dt * domega_new

        # orientation
        R_delta = rotation_matrix_from_vector(dt * omega_new)
        R_new = R @ R_delta

        # update time
        t_new = t + dt

        return state.replace(
            p=p_new,
            R=R_new,
            v=v_new,
            omega=omega_new,
            domega=domega_new,
            motor_omega=motor_omega_new,
            acc=acc_new,
            jrk=jrk,
            u=u,
            t=t_new,
        )

    def moment_map(self, motor_omega):
        """creates differentiable moment map from current moment omegas"""
        s = sigmoid(motor_omega, self._motor_xi)
        return s * self._ms_pos + (1.0 - s) * self._ms_neg

    def hovering_speed_map(self, g_bz):
        """
        creates differentiable motor hovering speed map from current
        projected gravity vector
        """
        s = sigmoid(g_bz, self._g_bz_xi)
        return (
            s * self.inverted_hovering_motor_speed
            + (1.0 - s) * self.hovering_motor_speed
        )

    def mixer_matrix(self, motor_omega):
        """creates differentiable mixer matrix from current motor state"""
        rotor_coordinates = jnp.stack(
            [self._tbm_fr, self._tbm_bl, self._tbm_fl, self._tbm_br]
        )
        x = rotor_coordinates[:, 0]
        y = rotor_coordinates[:, 1]
        ms = self.moment_map(motor_omega=motor_omega)

        return jnp.array(
            [jnp.ones(4), y, -x, ms * -self._motor_directions],
            dtype=jnp.float32,
        )


if __name__ == "__main__":
    quad = Quadrotor(mass=1.0)
    state = quad.default_state()
    ref_state = quad.default_state()
    control_state = quad._reference_controller.create_control_state()
    action = jnp.array([0.0, 0.0, 0.0, 1.0])
    eta = jnp.int32(1)
    dt = 0.1
    state_new = quad.step(state, control_state, ref_state, eta, action, dt)
    print(state_new)
