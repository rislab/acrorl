import os
import jax
import numpy as np
from .math import vee
from matplotlib import pyplot as plt
from flightning import FLIGHTNING_PATH
from flightning.envs.env_base import EnvTransition
from flightning.envs.quad_env import QuadEnvState


def _extract_traj_data(traj: EnvTransition):
    assert traj.reward.ndim == 2
    num_trajs = traj.reward.shape[0]
    state: QuadEnvState = traj.state
    done = np.logical_or(traj.terminated, traj.truncated)
    plt.rcParams["axes.grid"] = True

    fig1, axes1 = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), constrained_layout=True)
    fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), constrained_layout=True)
    fig3, axes3 = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), constrained_layout=True)
    fig4, axes4 = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), constrained_layout=True)
    fig5, axes5 = plt.subplots(nrows=4, ncols=1, figsize=(8, 9), constrained_layout=True, sharex=True)
    fig6, axes6 = plt.subplots(nrows=4, ncols=1, figsize=(8, 9), constrained_layout=True, sharex=True)
    fig7, axes7 = plt.subplots(nrows=4, ncols=1, figsize=(8, 9), constrained_layout=True, sharex=True)

    ax_px, ax_py, ax_pz = axes1
    ax_vx, ax_vy, ax_vz = axes2
    ax_Rx, ax_Ry, ax_Rz = axes3
    ax_wx, ax_wy, ax_wz = axes4
    ax_u1, ax_u2, ax_u3, ax_u4 = axes5
    ax_a1, ax_a2, ax_a3, ax_a4 = axes6

    pos_rows, vel_rows, bodyrate_rows = [], [], []
    attitude_rows, control_rows, action_rows, motor_rows = [], [], [], []

    for i in range(num_trajs):
        idx = np.where(done[i])[0][0].item() + 1

        t = state.time[i, :idx]
        x = state.quadrotor_state.p[i, :idx, 0]
        y = state.quadrotor_state.p[i, :idx, 1]
        z = state.quadrotor_state.p[i, :idx, 2]
        R = state.quadrotor_state.R[i, :idx]
        Rv = jax.vmap(vee)(R)
        Rx, Ry, Rz = Rv[:, 0], Rv[:, 1], Rv[:, 2]
        wx = state.quadrotor_state.omega[i, :idx, 0]
        wy = state.quadrotor_state.omega[i, :idx, 1]
        wz = state.quadrotor_state.omega[i, :idx, 2]
        vx = state.quadrotor_state.v[i, :idx, 0]
        vy = state.quadrotor_state.v[i, :idx, 1]
        vz = state.quadrotor_state.v[i, :idx, 2]
        motor_omega = state.quadrotor_state.motor_omega[i, :idx]
        u1 = state.quadrotor_state.u[i, :idx, 0]
        u2 = state.quadrotor_state.u[i, :idx, 1]
        u3 = state.quadrotor_state.u[i, :idx, 2]
        u4 = state.quadrotor_state.u[i, :idx, 3]
        actions = state.last_actions[i, :idx, -1, :]
        a1, a2, a3, a4 = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3]

        traj_id = np.full(t.shape, i, dtype=int)
        pos_rows.append(np.column_stack([traj_id, t, x, y, z]))
        vel_rows.append(np.column_stack([traj_id, t, vx, vy, vz]))
        bodyrate_rows.append(np.column_stack([traj_id, t, wx, wy, wz]))
        attitude_rows.append(np.column_stack([traj_id, t, Rx, Ry, Rz]))
        control_rows.append(np.column_stack([traj_id, t, u1, u2, u3, u4]))
        action_rows.append(np.column_stack([traj_id, t, a1, a2, a3, a4]))
        motor_rows.append(np.column_stack([traj_id, t, motor_omega]))

        ax_px.plot(t, x); ax_py.plot(t, y); ax_pz.plot(t, z)
        ax_vx.plot(t, vx); ax_vy.plot(t, vy); ax_vz.plot(t, vz)
        ax_wx.plot(t, wx); ax_wy.plot(t, wy); ax_wz.plot(t, wz)
        ax_Rx.plot(t, Rx); ax_Ry.plot(t, Ry); ax_Rz.plot(t, Rz)
        ax_u1.plot(t, u1); ax_u2.plot(t, u2); ax_u3.plot(t, u3); ax_u4.plot(t, u4)
        ax_a1.plot(t, a1); ax_a2.plot(t, a2); ax_a3.plot(t, a3); ax_a4.plot(t, a4)
        for m in range(4):
            axes7[m].plot(t, motor_omega[:, m])
            axes7[m].set_ylabel(rf"$\Omega_{m+1}$ [rad/s]")

    # labels
    fig1.suptitle("Quadrotor Position")
    for ax in axes1: ax.set_xlabel("Time, $t$ [s]")
    ax_px.set_ylabel("$x$ [m]"); ax_py.set_ylabel("$y$ [m]"); ax_pz.set_ylabel("$z$ [m]")

    fig2.suptitle("Quadrotor Linear Velocity")
    for ax in axes2: ax.set_xlabel("Time, $t$ [s]")
    ax_vx.set_ylabel("$v_x$ [m/s]"); ax_vy.set_ylabel("$v_y$ [m/s]"); ax_vz.set_ylabel("$v_z$ [m/s]")

    fig3.suptitle("Quadrotor Orientation")
    for ax in axes3: ax.set_xlabel("Time, $t$ [s]")
    ax_Rx.set_ylabel("$R_x$"); ax_Ry.set_ylabel("$R_y$"); ax_Rz.set_ylabel("$R_z$")

    fig4.suptitle("Quadrotor Angular Velocity")
    for ax in axes4: ax.set_xlabel("Time, $t$ [s]")
    ax_wx.set_ylabel("$\omega_x$ [rad/s]"); ax_wy.set_ylabel("$\omega_y$ [rad/s]"); ax_wz.set_ylabel("$\omega_z$ [rad/s]")

    fig5.suptitle("Control Action, $u$")
    for ax in axes5: ax.set_xlabel("Time, $t$ [s]")
    ax_u1.set_ylabel("$f_c$ [N]"); ax_u2.set_ylabel(r"$\tau_x$ [Nm]")
    ax_u3.set_ylabel(r"$\tau_y$ [Nm]"); ax_u4.set_ylabel(r"$\tau_z$ [Nm]")

    fig6.suptitle("Learned Actions")
    for ax in axes6: ax.set_xlabel("Time, $t$ [s]")
    ax_a1.set_ylabel(r"$a_{x}$ [m]"); ax_a2.set_ylabel(r"$a_{y}$ [m]")
    ax_a3.set_ylabel(r"$a_{z}$ [m]"); ax_a4.set_ylabel(r"$a_{\eta}$ [thrust posture]")

    fig7.suptitle("Quadrotor Motor Rates")
    for ax in axes7: ax.set_xlabel("Time, $t$ [s]")

    figs = (fig1, fig2, fig3, fig4, fig5, fig6, fig7)
    data = {
        "pos": np.vstack(pos_rows),
        "vel": np.vstack(vel_rows),
        "bodyrate": np.vstack(bodyrate_rows),
        "attitude": np.vstack(attitude_rows),
        "controls": np.vstack(control_rows),
        "actions": np.vstack(action_rows),
        "motors": np.vstack(motor_rows),
    }
    return figs, data


def plot_trajectories(traj: EnvTransition):
    figs, _ = _extract_traj_data(traj)
    return figs


def save_trajectories(traj: EnvTransition, trial_name: str, save_plots: bool = True, save_data: bool = True):
    figs, data = _extract_traj_data(traj)

    fig_names = ("position", "velocity", "attitude", "bodyrate", "controls", "actions", "motors")
    csv_headers = {
        "pos":      "traj_id,t,x,y,z",
        "vel":      "traj_id,t,vx,vy,vz",
        "bodyrate": "traj_id,t,wx,wy,wz",
        "attitude": "traj_id,t,Rx,Ry,Rz",
        "controls": "traj_id,t,u1,u2,u3,u4",
        "actions":  "traj_id,t,a1,a2,a3,a4",
        "motors":   "traj_id,t,omega1,omega2,omega3,omega4",
    }

    if save_plots:
        os.makedirs("plots", exist_ok=True)
        for fig, name in zip(figs, fig_names):
            fig.savefig(f"plots/{trial_name}_{name}.png")
        plots_path = f"{FLIGHTNING_PATH}/../plots"
        print(f"[PLOTTING] Plots saved: {plots_path}")
        

    if save_data:
        os.makedirs("data", exist_ok=True)
        for key, header in csv_headers.items():
            np.savetxt(
                f"data/{trial_name}_{key}.csv",
                data[key],
                delimiter=",",
                header=header,
                comments="",
            )
        data_path = f"{FLIGHTNING_PATH}/../data"
        print(f"[PLOTTING] Data saved: {data_path}")