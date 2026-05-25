import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import patheffects as pe
import matplotlib.transforms as mtransforms
from pathlib import Path

from .uav import Uav
from flightning.envs.env_base import EnvTransition, EnvState


class QuadrotorAnimator:
    def __init__(
        self,
        goal_x,
        goal_R,
        drone_scale,
        draw_trajectories,
        dpi,
        bitrate,
        render_hz,
    ):
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "font.size": 12,
                "axes.titlesize": 16,
                "axes.labelsize": 16,
                "axes.grid": True,
                "lines.linewidth": 1.5,
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")

        # tick params
        self.ax.tick_params(axis="both")
        self.ax.tick_params(axis="z")
        self.ax.tick_params(pad=2)

        # axis labels
        self.ax.set_xlabel("$x$ (m)")
        self.ax.set_ylabel("$y$ (m)")
        self.ax.set_zlabel("$z$ (m)")

        # projection & aspect ratio
        self.ax.set_proj_type("ortho")
        self.ax.set_box_aspect((1, 1, 1))

        # settings
        self.dpi = dpi
        self.bitrate = bitrate
        self.render_hz = render_hz
        self.drone_scale = drone_scale
        self.draw_trajectories = draw_trajectories

        # goal drone
        self.goal_x = np.asarray(goal_x)
        self.goal_R = np.asarray(goal_R)

    def animate_trajectories(
        self,
        traj: EnvTransition,
        title: str = "Quadrotor Animation",
        trial_name: str = None,
        extension: str = None,
        save_animation: bool = False,
    ):
        """
        Animate and optionally save trajectories.
        """
        num_trajs = traj.reward.shape[0]
        state = traj.state
        uavs = [Uav(self.ax, scale=self.drone_scale) for _ in range(num_trajs)]

        # first termination index (per env) + 1
        idxs = np.zeros(num_trajs, dtype=np.int32)
        done = np.logical_or(traj.terminated, traj.truncated)
        for i in range(num_trajs):
            idxs[i] = int(np.where(done[i])[0][0]) + 1
        n_frames = int(np.max(idxs))

        # extract trajectories position and orientation (num_trajs, n_frames, 3/3x3)
        x = np.zeros((num_trajs, n_frames, 3))
        R = np.zeros((num_trajs, n_frames, 3, 3))
        for j in range(num_trajs):
            idx = int(idxs[j])
            x[j, :idx, :] = state.quadrotor_state.p[j, :idx, 0:3]
            R[j, :idx, :] = state.quadrotor_state.R[j, :idx]

        # setting x, y, z limits
        self.ax.set_xlim([-1.05 ,1.05])
        self.ax.set_ylim([-1.05, 1.05])
        self.ax.set_zlim([-1.05, 1.05])
        self.ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        self.ax.zaxis.set_major_locator(plt.MultipleLocator(0.5))

        # timing / downsampling
        dt_sim = float(state.time[0, 1] - state.time[0, 0])
        dt_render = 1.0 / self.render_hz
        stride = max(1, int(round(dt_render / dt_sim)))
        frame_idxs = range(0, n_frames, stride)
        interval_ms = 1000.0 / self.render_hz
        fps = max(1, int(round(self.render_hz)))

        # run animation
        self.fig.suptitle(title, y=0.98)
        animation = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=frame_idxs,
            fargs=(x, R, uavs),
            interval=interval_ms,
            blit=False,
        )

        # save animation depending on extension type
        if save_animation:
            if extension in {"mp4", "m4v", "mov"}:
                os.makedirs("videos/", exist_ok=True)
                filename = "videos/" + trial_name
                writer = FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    bitrate=self.bitrate,
                    extra_args=["-pix_fmt", "yuv420p"],
                )
            elif extension == "gif":
                os.makedirs("gifs/", exist_ok=True)
                filename = "gifs/" + trial_name
                writer = PillowWriter(fps=fps)
            path = Path(filename + "." + extension)
            animation.save(str(path), writer=writer, dpi=self.dpi)
            print(f"[PyPlot3D] Animation saved: {path}")

    def update_plot(self, i, x, R, uavs):
        for k, uav in enumerate(uavs):
            uav.draw_at(x[k, i, :], R[k, i, :, :])
            if self.draw_trajectories:
                self.ax.plot(
                    x[k, :i, 0],
                    x[k, :i, 1],
                    x[k, :i, 2],
                    color="goldenrod",
                    linewidth=0.5,
                    alpha=0.2,
                )
        return []

    @staticmethod
    def _select_pose_indices(frame_start: int, frame_stop: int, num_poses: int):
        if frame_stop <= frame_start:
            return np.array([frame_start], dtype=int)
        idxs = np.linspace(frame_start, frame_stop, num=num_poses)
        return np.unique(np.round(idxs).astype(int))

    @staticmethod
    def _select_pose_indices_by_speed(
        frame_start: int, frame_stop: int, x_traj: np.ndarray, num_poses: int
    ):
        """
        Sample pose indices so that faster-moving segments get more frames.
        """
        seg = x_traj[frame_start : frame_stop + 1]
        diffs = np.diff(seg, axis=0)
        step_lengths = np.linalg.norm(diffs, axis=1)
        arc = np.concatenate([[0.0], np.cumsum(step_lengths)])

        total = arc[-1]
        if total < 1e-6:
            return QuadrotorAnimator._select_pose_indices(
                frame_start, frame_stop, num_poses
            )

        target_arcs = np.linspace(0, total, num_poses)
        local_idxs = np.searchsorted(arc, target_arcs)
        local_idxs = np.clip(local_idxs, 0, len(arc) - 1)
        return np.unique(local_idxs + frame_start).astype(int)

    def plot_still_frames(
        self,
        traj: EnvTransition,
        trial_name: str = None,
        title: str = "Quadrotor Flip Still Frames",
        drone_scale: float = 0.1,
        num_poses: int = 7,
        flip_window: tuple = (0, 110),
        colorbar: bool = True,
        save_still_frames: bool = False,
    ):
        """
        Plot still frames with motor rate colour-coding.
        Motors are coloured on a blue-->red colormap normalised to
        """
        assert traj.reward.ndim == 2
        num_trajs = traj.reward.shape[0]
        state: EnvState = traj.state

        # first termination index (per env) + 1
        idxs = np.zeros(num_trajs, dtype=np.int32)
        done = np.logical_or(traj.terminated, traj.truncated)
        for i in range(num_trajs):
            idxs[i] = int(np.where(done[i])[0][0]) + 1
        n_frames = int(np.max(idxs))

        # (num_trajs, n_frames, 3 / 3x3 / 4)
        x = np.zeros((num_trajs, n_frames, 3))
        R = np.zeros((num_trajs, n_frames, 3, 3))
        motor_omega = np.zeros((num_trajs, n_frames, 4))
        for j in range(num_trajs):
            idx = int(idxs[j])
            x[j, :idx, :] = state.quadrotor_state.p[j, :idx, 0:3]
            R[j, :idx, :, :] = state.quadrotor_state.R[j, :idx]
            motor_omega[j, :idx, :] = state.quadrotor_state.motor_omega[j, :idx, 0:4]

        # motor rate normalization bounds
        omega_min = float(np.min(-2200.0))
        omega_max = float(np.max(2200.0))
        cmap = plt.get_cmap("coolwarm")
        norm = plt.Normalize(vmin=omega_min, vmax=omega_max)

        # evenly-spaced frame indices inside the flip window
        f_start = max(0, flip_window[0])
        f_stop = min(n_frames - 1, flip_window[1])
        pose_idxs = self._select_pose_indices_by_speed(f_start, f_stop, x[0], num_poses)

        # X, Y, Z limis
        traj_slice = x[0, f_start : f_stop + 1, :]
        center = traj_slice.mean(axis=0)
        half = max((traj_slice.max(axis=0) - traj_slice.min(axis=0)).max() / 2.0, 0.3)
        pad = 0.25
        self.ax.set_xlim([center[0] - pad, center[0] + pad])
        self.ax.set_ylim([center[1] - pad, center[1] + pad])
        self.ax.set_zlim([center[2] - pad, center[2] + pad])
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        self.ax.zaxis.set_major_locator(plt.MultipleLocator(0.1))

        for k in range(num_trajs):
            # one Uav per pose so each gets its own independent artists
            pose_uavs = [Uav(self.ax, scale=drone_scale) for _ in pose_idxs]

            for uav, frame_i in zip(pose_uavs, pose_idxs):
                omegas = motor_omega[k, frame_i, :]
                motor_colors = [cmap(norm(w)) for w in omegas]
                uav.draw_at(
                    x=x[k, frame_i, :], R=R[k, frame_i, :, :], motor_colors=motor_colors
                )

                if self.draw_trajectories and frame_i > f_start:
                    # colored trajectory segment up to this pose
                    seg_x = x[k, f_start : frame_i + 1, 0]
                    seg_y = x[k, f_start : frame_i + 1, 1]
                    seg_z = x[k, f_start : frame_i + 1, 2]
                    points = np.stack([seg_x, seg_y, seg_z], axis=-1)
                    segments = np.stack([points[:-1], points[1:]], axis=1)
                    seg_frames = np.arange(f_start, frame_i)          # frame index per segment
                    avg_omega = motor_omega[k, seg_frames, :].mean(axis=1)  # mean over 4 rotors
                    lc = Line3DCollection(
                        segments,
                        cmap="coolwarm",
                        linewidth=1.5,
                        alpha=0.3,
                        antialiased=True,
                    )
                    lc.set_array(avg_omega)
                    lc.set_clim(omega_min, omega_max)
                    self.ax.add_collection3d(lc)

        # colorbar
        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = self.fig.colorbar(sm, ax=self.ax, shrink=0.55, pad=0.1)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(r"$\boldsymbol{\Omega}$ (rad/s)", fontsize=14)
            pad_right = 0.2
            x_title = 0.425
        else:
            pad_right = 0.6
            x_title = 0.5

        self.ax.view_init(elev=20, azim=-30)
        # self.fig.suptitle(title, x=x_title, y=0.95)  #

        if save_still_frames is not None:
            os.makedirs("plots", exist_ok=True)
            path = Path("plots/" + trial_name + "_still_frames.png")
            renderer = self.fig.canvas.get_renderer()
            tight_bbox = self.fig.get_tightbbox(renderer)
            expanded_bbox = mtransforms.Bbox.from_extents(
                tight_bbox.x0 - 0.1,
                tight_bbox.y0 - 0.2,
                tight_bbox.x1 + pad_right,
                tight_bbox.y1 - 0.1,
            )

            self.fig.savefig(str(path), dpi=self.dpi, bbox_inches=expanded_bbox)
            print(f"[PyPlot3D] Still frames saved: {path}")
