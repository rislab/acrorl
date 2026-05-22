import os
import jax
import jax.numpy as jnp
import numpy as np
from flightning.envs.env_base import EnvTransition
from flightning.envs.quad_env import QuadEnvState
from .math import proj_gravity


def eval_trajectories(
    traj: EnvTransition, trial_name: str, goal_g_b: jnp.ndarray, goal_pos: jnp.ndarray, save_data: bool
):
    if save_data:
        os.makedirs("data", exist_ok=True)
        data_filename = "data/" + trial_name
    else:
        data_filename = None
    assert traj.reward.ndim == 2
    num_trajs, horizon = traj.reward.shape

    done = jnp.logical_or(traj.terminated, traj.truncated)
    state: QuadEnvState = traj.state

    ep_len = jnp.argmax(done, axis=1) + 1
    step_idx = jnp.arange(horizon)
    valid = step_idx[None, :] < ep_len[:, None]
    p_all = state.quadrotor_state.p
    R_all = state.quadrotor_state.R
    t_all = state.time
    goal_g_b_all = goal_g_b

    # --- position rmse ---
    pos_err_all = jnp.linalg.norm(p_all - goal_pos, axis=-1)
    pos_rmse = jnp.sqrt(
        jnp.sum(pos_err_all**2 * valid, axis=1)
        / jnp.maximum(ep_len, 1).astype(jnp.float32)
    )

    # --- settling time ---
    g_b_all = jax.vmap(jax.vmap(proj_gravity))(R_all)
    cos_angle = jnp.clip(jnp.sum(g_b_all * goal_g_b_all, axis=-1), -1.0, 1.0)
    g_ang_all = jnp.arccos(cos_angle)
    g_ang_thresh = jnp.deg2rad(10.0)
    below = g_ang_all < g_ang_thresh
    suffix_all = (
        jnp.flip(jnp.cumsum(jnp.flip(~below & valid, axis=1), axis=1), axis=1) == 0
    )
    settled_mask = suffix_all & valid
    settled = jnp.any(settled_mask, axis=1)
    settle_step = jnp.argmax(settled_mask, axis=1)
    final_idx = ep_len - 1
    settle_time = jnp.where(
        settled,
        t_all[jnp.arange(num_trajs), settle_step],
        t_all[jnp.arange(num_trajs), final_idx],
    )

    # --- max position deviation from 0 in each axis ---
    x_all = p_all[:, :, 0]
    y_all = p_all[:, :, 1]
    z_all = p_all[:, :, 2]
    max_x_dev = jnp.max(jnp.abs(x_all) * valid, axis=1)
    max_y_dev = jnp.max(jnp.abs(y_all) * valid, axis=1)
    max_z_dev = jnp.max(jnp.abs(z_all) * valid, axis=1)

    # --- success ---
    crashed = jnp.any(traj.terminated & valid, axis=1)
    success = (settled & ~crashed).astype(jnp.float32)

    metrics = {
        "pos_rmse": pos_rmse,
        "settle_time": settle_time,
        "max_x_dev": max_x_dev,
        "max_y_dev": max_y_dev,
        "max_z_dev": max_z_dev,
        "success": success,
    }

    metrics = {
        "pos_rmse_mean": jnp.mean(pos_rmse),
        "settle_time_mean": jnp.mean(settle_time),
        "max_x_dev": jnp.max(max_x_dev),
        "max_y_dev": jnp.max(max_y_dev),
        "max_z_dev": jnp.max(max_z_dev),
        "success_rate": jnp.mean(success),
        "per_traj": metrics,
    }

    if data_filename is not None:

        pos_rmse_np = np.array(pos_rmse)
        settle_time_np = np.array(settle_time)
        max_x_dev_np = np.array(max_x_dev)
        max_y_dev_np = np.array(max_y_dev)
        max_z_dev_np = np.array(max_z_dev)
        success_np = np.array(success)

        per_traj_data = np.column_stack(
            [
                np.arange(num_trajs),
                pos_rmse_np,
                settle_time_np,
                max_x_dev_np,
                max_y_dev_np,
                max_z_dev_np,
                success_np,
            ]
        )

        np.savetxt(
            f"{data_filename}_metrics.csv",
            per_traj_data,
            delimiter=",",
            header="traj_id,pos_rmse,settle_time,max_x_dev,max_y_dev,max_z_dev,success",
            comments="",
        )

        # --- summary CSV ---
        summary_data = np.array(
            [
                metrics["pos_rmse_mean"],
                metrics["settle_time_mean"],
                metrics["max_x_dev"],
                metrics["max_y_dev"],
                metrics["max_z_dev"],
                metrics["success_rate"],
            ]
        )

        np.savetxt(
            f"{data_filename}_mean_metrics.csv",
            summary_data[None, :],
            delimiter=",",
            header="pos_rmse_mean,settle_time_mean,max_x_dev,max_y_dev,max_z_dev,success_rate",
            comments="",
        )

    return metrics
