import yaml
import functools
from typing import Optional

import chex
import jax
import jax_dataclasses as jdc
import numpy as np
from jax import numpy as jnp
import jax.scipy.spatial.transform as transform


from flightning.objects import Quadrotor, QuadrotorState, WorldBox
from flightning.controllers import HopfControllerState
from flightning.trajectories import CompositeTrajectory, TRAJECTORY_REGISTRY
from flightning.utils import math as math_utils
from flightning.utils import spaces
from flightning.utils.pytrees import stack_pytrees
from flightning.utils.math import proj_gravity, smooth_l1, l2_penalty, l1
from flightning.utils.random import key_generator, random_rotation
import flightning.envs.env_base as env_base
from flightning.envs.env_base import EnvTransition


@jdc.pytree_dataclass
class QuadEnvState(env_base.EnvState):
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    control_state: HopfControllerState
    last_actions: jax.Array
    last_quadrotor_states: QuadrotorState
    trajectory: CompositeTrajectory

    # last_quadrotor_states: QuadrotorState
    # queues last actions to handle delay
    # last_actions[-1] is the most recent action
    # last_actions[0] is the oldest action


class QuadEnv(env_base.Env[QuadEnvState]):
    """State-based quadrotor environment."""

    def __init__(
        self,
        *,
        max_steps_in_episode=10000,
        trajectory_profile="constant_reference",
        trajectory_path=None,
        dt=0.02,
        delay=0.01,
        roll_pitch_range=0.1,
        yaw_range=jnp.pi,
        velocity_std=0.1,
        omega_std=0.1,
        margin=1.5,
        drone_path=None,
        start_inverted=False,
        randomize_reset=True,
        action_history=3,
        num_last_quad_states=10
    ):
        # env params
        self.world_box = WorldBox(
            jnp.array([-5.0, -5.0, -5.0]), jnp.array([5.0, 5.0, 5.0])
        )
        self.goal_pos = 0.5 * (self.world_box.min + self.world_box.max)
        self.goal_g_b = jax.lax.select(
            start_inverted, jnp.array([0.0, 0.0, -1.0]), jnp.array([0.0, 0.0, 1.0])
        )
        self.margin = margin
        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)
        self.duration = max_steps_in_episode * dt
        assert delay >= 0.0, "Delay must be non-negative"
        self.delay = np.array(delay)
        self.action_history = action_history

        # random state params
        self.roll_pitch_range = roll_pitch_range
        self.yaw_range = yaw_range
        self.velocity_std = velocity_std
        self.omega_std = omega_std
        self.randomize_reset = randomize_reset
        self.R_inv = jax.lax.select(
            start_inverted, jnp.diag(jnp.array([1.0, -1.0, -1.0])), jnp.eye(3)
        )
        self.init_chart = jax.lax.select(start_inverted, jnp.int32(2), jnp.int32(1))
        self.init_eta = jax.lax.select(start_inverted, jnp.int32(-1), jnp.int32(1))
        itn_weights = jnp.array(
            [
                5.000,  # position
                3.000,  # orientation
                0.100,  # thrust posture
                0.000,  # velocity
                0.750,  # angular velocity
                0.250,  # action rate
            ]
        )
        nti_weights = jnp.array(
            [
                5.000,  # position
                3.000,  # orientation
                0.100,  # thrust posture
                0.005,  # velocity
                0.200,  # angular velocity
                0.200,  # action rate
            ]
        )
        self.weights = jax.lax.select(start_inverted, itn_weights, nti_weights)

        # quadrotor
        if drone_path:
            self.quadrotor = Quadrotor.from_yaml(drone_path)
        else:
            self.quadrotor = Quadrotor()

        # trajectory
        with open(trajectory_path, "r") as file:
            all_profiles = yaml.safe_load(file)
        profile_dict = all_profiles[trajectory_profile]

        self.traj_sequence = []
        for key in sorted(profile_dict.keys()):
            params = profile_dict[key].copy()
            if params.get("duration") == "env_duration":
                params["duration"] = self.duration
            else:
                params["duration"] = float(params["duration"])
            self.traj_sequence.append(params)

        # action limits
        self.action_min = self.quadrotor._action_min
        self.action_max = self.quadrotor._action_max

        # observation limits
        self.v_min = jnp.array([-10.0, -10.0, -10.0])
        self.v_max = jnp.array([10.0, 10.0, 10.0])
        self.omega_min = jnp.array([-30.0, -30.0, -30.0])
        self.omega_max = jnp.array([30.0, 30.0, 30.0])
        self.motor_omega_max = self.quadrotor._motor_omega_max
        self.motor_omega_min = self.quadrotor._motor_omega_min

        # inital action & history
        self.hovering_action = jnp.zeros_like(self.quadrotor._action_min)
        self.num_last_quad_states = num_last_quad_states

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key, state: Optional[QuadEnvState] = None
    ) -> tuple[QuadEnvState, jax.Array]:
        """resetting the env"""
        # random state
        key_p, key_R, key_v, key_omega, key_dr = jax.random.split(key, 5)
        if self.randomize_reset:
            p = jax.random.uniform(
                key_p,
                shape=(3,),
                minval=self.world_box.min + self.margin,
                maxval=self.world_box.max - self.margin,
            )
            rot = random_rotation(
                key_R,
                yaw_range=self.yaw_range,
                pitch_range=self.roll_pitch_range,
                roll_range=self.roll_pitch_range,
            )
            # rot = random_rotation_vector_limited(key_R, self.angle_limit)
            R = rot.as_matrix()
            yaw = rot.as_euler("ZYX")[0]
            v = self.velocity_std * jax.random.normal(key_v, shape=(3,))
            omega = self.omega_std * jax.random.normal(key_omega, shape=(3,))
        else:
            p = jnp.zeros(3)
            R = jnp.eye(3)
            v = jnp.zeros(3)
            omega = jnp.zeros(3)
            yaw = 0.0
        R = R @ self.R_inv

        # create state
        quadrotor_state = self.quadrotor.create_state(
            p=p, R=R, v=v, omega=omega, dr_key=key_dr
        )

        # create trajectory
        trajectories = []
        for params in self.traj_sequence:
            kwargs = dict(params)

            # setting initial state for optimization
            kwargs["init_quadrotor_state"] = quadrotor_state
            kwargs["init_yaw"] = yaw

            traj_type = kwargs.pop("type")
            if traj_type == "constant" or traj_type == "polynomial":
                flat_target = kwargs.pop("flat_target")
                p_target = jnp.array(flat_target[0:3])
                rotation_target = transform.Rotation.from_euler(
                    "zyx", jnp.array([flat_target[3], 0.0, 0.0])
                )
                target_quadrotor_state = self.quadrotor.create_state(
                    p=p_target,
                    R=rotation_target.as_matrix(),
                    v=jnp.zeros(3),
                    omega=jnp.zeros(3),
                    dr_key=key_dr,
                )
                kwargs["target_quadrotor_state"] = target_quadrotor_state

            traj_class = TRAJECTORY_REGISTRY[traj_type]
            trajectories.append(traj_class(**kwargs))
        trajectory = CompositeTrajectory(trajectories=trajectories)

        # create control state
        control_state = self.quadrotor._reference_controller.create_control_state(
            chart=self.init_chart, eta=self.init_eta,
        )

        # title empty last actions and states
        last_actions = jnp.tile(self.hovering_action, (self.action_history, 1))
        last_quadrotor_states = [quadrotor_state] * self.num_last_quad_states
        last_quadrotor_states = stack_pytrees(last_quadrotor_states)
        state = QuadEnvState(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            control_state=control_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quadrotor_states,
            trajectory=trajectory,
        )
        obs = self._get_obs(state)
        return state, obs

    def _get_obs(self, state: QuadEnvState) -> jax.Array:
        """get observations from env"""
        return jnp.concatenate(
            [
                state.quadrotor_state.p,
                math_utils.vec(state.quadrotor_state.R),
                state.quadrotor_state.v,
                state.quadrotor_state.omega,
                state.last_actions.flatten(),
            ]
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: QuadEnvState, action: jax.Array, key: chex.PRNGKey
    ) -> EnvTransition:
        # clip action
        action = jnp.clip(action, self.action_space.low, self.action_space.high)

        # add action to last actions
        last_actions = jnp.roll(state.last_actions, shift=-1, axis=0)
        last_actions = last_actions.at[-1].set(action)

        # steps
        dt_1 = self.delay % self.dt
        dt_2 = self.dt - dt_1

        # actions
        action_1 = last_actions[0]
        action_2 = last_actions[1]

        # state
        quadrotor_state = state.quadrotor_state
        control_state = state.control_state

        reference_quadrotor_state, reference_yaw, reference_dyaw, reference_eta = (
            state.trajectory.get_reference(state.time)
        )

        # stepping dynamics (accounting for delay if present)
        quadrotor_state, control_state = jax.lax.cond(
            dt_1 > 0.0,
            lambda carry: self.quadrotor.step(
                quadrotor_state=carry[0],
                control_state=carry[1],
                reference_quadrotor_state=reference_quadrotor_state,
                reference_yaw=reference_yaw,
                reference_dyaw=reference_dyaw,
                reference_eta=reference_eta,
                action=action_1,
                dt=dt_1,
            ),
            lambda carry: carry,
            (quadrotor_state, control_state),
        )

        quadrotor_state, control_state = self.quadrotor.step(
            quadrotor_state=quadrotor_state,
            control_state=control_state,
            reference_quadrotor_state=reference_quadrotor_state,
            reference_yaw=reference_yaw,
            reference_dyaw=reference_dyaw,
            reference_eta=reference_eta,
            action=action_2,
            dt=dt_2,
        )

        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            control_state=control_state,
            last_actions=last_actions,
        )

        obs = self._get_obs(next_state)
        reward = self._get_reward(state, next_state, key)
        terminated = self._is_colliding(next_state)
        truncated = jnp.greater_equal(next_state.step_idx, self.max_steps_in_episode)
        return EnvTransition(next_state, obs, reward, terminated, truncated, dict())

    def _get_reward(
        self, last_state: QuadEnvState, next_state: QuadEnvState, key: chex.PRNGKey
    ) -> jax.Array:
        # from state
        action = next_state.last_actions[-1]
        last_action = last_state.last_actions[-1]
        p = next_state.quadrotor_state.p
        v = next_state.quadrotor_state.v
        omega = next_state.quadrotor_state.omega
        R = next_state.quadrotor_state.R
        g_b = proj_gravity(R)
        time_left = self.max_steps_in_episode - next_state.step_idx
        crashed = self._is_colliding(next_state)

        # error signals
        pos_err = p - jnp.zeros_like(p)
        vel_err = v - jnp.zeros_like(v)
        omega_err = omega - jnp.zeros_like(omega)
        g_b_err = g_b - self.goal_g_b
        eta_err = 1.0 - action[-1] ** 2
        signals = jnp.array(
            [
                l1(pos_err),
                l2_penalty(g_b_err),
                eta_err,
                smooth_l1(vel_err),
                smooth_l1(omega_err),
                l2_penalty(action[0:3] - last_action[0:3]),
            ]
        )

        # cost
        step_c = jnp.dot(signals, self.weights)
        collision_cost = jax.lax.select(crashed, time_left * step_c, 0.0)
        step_c += jax.lax.stop_gradient(collision_cost)
        return -step_c * self.dt

    def _is_colliding(self, state: QuadEnvState) -> jax.Array:
        quad_state = state.quadrotor_state
        world_box = self.world_box
        world_collision = jnp.logical_not(world_box.contains(quad_state.p))
        is_colliding = world_collision
        return is_colliding

    def get_world_constraints(self) -> jax.Array:
        return self.world_box.min, self.world_box.max

    def get_goal(self) -> jax.Array:
        return self.goal_pos, self.goal_g_b

    @property
    def action_space(self) -> spaces.Box:
        """defines action space and limits"""
        low = self.action_min
        high = self.action_max
        return spaces.Box(low, high, shape=self.action_min.shape)

    @property
    def observation_space(self) -> spaces.Box:
        """defines observation space and limits"""
        n = self.action_history
        action_high_repeated = jnp.concatenate([self.action_space.high] * n)
        action_low_repeated = jnp.concatenate([self.action_space.low] * n)

        return spaces.Box(
            low=jnp.concatenate(
                [
                    self.world_box.min,
                    -jnp.ones(9),
                    self.v_min,
                    self.omega_min,
                    action_low_repeated,
                ]
            ),
            high=jnp.concatenate(
                [
                    self.world_box.max,
                    jnp.ones(9),
                    self.v_max,
                    self.omega_max,
                    action_high_repeated,
                ]
            ),
            shape=(
                jnp.ones(3).shape
                + jnp.ones(9).shape
                + self.v_min.shape
                + self.omega_min.shape
                + action_low_repeated.shape
            ),
        )


if __name__ == "__main__":
    key_gen = key_generator(0)

    env = QuadEnv()

    state, *_ = env.reset(next(key_gen))
    random_action = env.action_space.sample(next(key_gen))
    state, obs, *_ = env.step(state, random_action, next(key_gen))
    print(obs)
