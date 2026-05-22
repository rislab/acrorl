import argparse
import jax
import jax.numpy as jnp
import jax.scipy.spatial.transform as transform
import matplotlib.pyplot as plt

from flightning import FLIGHTNING_PATH
from flightning.envs import QuadEnv, rollout
from flightning.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
from flightning.animation import QuadrotorAnimator
from flightning.utils.printing import print_metrics
from flightning.utils.plotting import plot_trajectories, save_trajectories
from flightning.utils.evaluation import eval_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline inversion strategies.")

    parser.add_argument("--inversion_method", type=str, choices=["step", "step_oca", "min_snap", "min_snap_oca"], required=True, 
                        help="inversion type: 'nti' (nominal to inverted) or 'itn' (inverted to nominal).")
    parser.add_argument("--inversion_type", type=str, choices=["nti", "itn"], required=True,
                        help="inversion type: 'nti' (nominal to inverted) or 'itn' (inverted to nominal).")
    parser.add_argument("--num_drones", type=int, default=1,
                        help="number of parallel rollouts")
    parser.add_argument("--duration", type=int, default=5,
                        help="episode duration in seconds")
    parser.add_argument("--margin", type=float, default=4.9,
                        help="margin for start position from env limits")
    parser.add_argument("--randomize_reset", action="store_true",
                        help="randomize samples from inside margin, else from 0")
    parser.add_argument("--drone_name", type=str, default="eris03_eval",
                        help="drone yaml filename (without .yaml extension)")
    parser.add_argument("--save_plots", action="store_true",
                        help="save trajectory plots as PNGs to disk")
    parser.add_argument("--save_data", action="store_true",
                        help="save trajectory data as CSVs to disk")
    parser.add_argument("--save_animation", action="store_true",
                        help="save animation to disk")
    parser.add_argument("--save_still_frames", action="store_true",
                        help="save still frame PNG to disk")
    parser.add_argument("--trial_name", type=str, default=None,
                        help="override the trial name used for saving; "
                             "defaults to inversion_type if saving is enabled")
    parser.add_argument("--show_hud", action="store_true",
                        help="show all figures interactively after rendering")
    parser.add_argument("--animate", action="store_true",
                        help="render an animation of the trajectories")
    parser.add_argument("--extension", type=str, choices=["gif", "mp4"], default="gif",
                        help="animation format (default: gif)")
    parser.add_argument("--dpi", type=int, default=1000,
                        help="DPI for gif/png output (default: 1000)")
    parser.add_argument("--bitrate", type=int, default=100000,
                        help="bitrate for mp4 output (default: 100000)")
    parser.add_argument("--render_hz", type=int, default=50,
                        help="render frame rate in Hz (default: 50)")
    parser.add_argument("--drone_scale", type=float, default=0.25,
                        help="visual scale of the drone mesh (default: 0.25)")
    parser.add_argument("--render_still_frames", action="store_true",
                        help="render still-frame pose sequence")
    parser.add_argument("--num_poses", type=int, default=7,
                        help="number of poses in still-frame render (default: 7)")
    parser.add_argument("--flip_window", type=int, nargs=2, default=[0, 110],
                        metavar=("START", "END"),
                        help="frame window for the flip (default: 0 110)")
    parser.add_argument("--colorbar", action="store_true",
                        help="show colorbar in still-frame render")

    return parser.parse_args()


# -----------------------
# params
# -----------------------

seed = 0
dt = 0.02  # [s]
delay = 0.001  # [s]
velocity_std = 0.1  # [m/s]
roll_pitch_range = 0.1  # [rad]
yaw_range = jnp.pi  # [rad]
omega_std = 0.1  # [rad/s]
trajectory_path = FLIGHTNING_PATH + "/trajectories/config/trajectories.yaml"


def main():
    args = parse_args()

    if args.inversion_type[0:3] == "nti":
        start_inverted = False
        title_prefix = r"Nominal-to-inverted hover transition"
        traj_prefix = "nti"
    else:
        start_inverted = True
        title_prefix = r"Inverted-to-nominal hover transition"
        traj_prefix = "itn"

    if args.inversion_method == "min_snap":
        drone_name = args.drone_name 
        trajectory_profile = traj_prefix + "_min_snap"
        title_suffix = r"$\min \boldsymbol{r}^{(4)}{+}$HFCA"
        policy_eta = 1.0
    elif args.inversion_method == "min_snap_oca":
        drone_name = args.drone_name + "_oca"
        trajectory_profile = traj_prefix + "_min_snap"
        title_suffix = r"$\min \boldsymbol{r}^{(4)}{+}$HFCA${+}$OCA"
        policy_eta = 1.0
    elif args.inversion_method == "step":
        drone_name = args.drone_name
        trajectory_profile = "constant_reference"
        title_suffix = r"$S(t){+}$HFCA"
        policy_eta = jax.lax.select(start_inverted, 1.0, -1.0)
    elif args.inversion_method == "step_oca":
        drone_name = args.drone_name + "_oca"
        trajectory_profile = "constant_reference"
        title_suffix = r"$S(t){+}$HFCA${+}$OCA"
        policy_eta = jax.lax.select(start_inverted, 1.0, -1.0)

    animation_title = title_prefix + "\n" + title_suffix
    
    saving = args.save_plots or args.save_data
    trial_name = args.trial_name if args.trial_name is not None else args.inversion_type + "_" + args.inversion_method

    # -----------------------
    # env definition
    # -----------------------

    key = jax.random.key(seed)
    drone_path = f"{FLIGHTNING_PATH}/objects/quadrotor_files/{drone_name}.yaml"

    env = QuadEnv(
        max_steps_in_episode=args.duration * int(1 / dt),
        trajectory_profile=trajectory_profile,
        trajectory_path=trajectory_path,
        dt=dt,
        delay=delay,
        roll_pitch_range=roll_pitch_range,
        yaw_range=yaw_range,
        velocity_std=velocity_std,
        omega_std=omega_std,
        margin=args.margin,
        drone_path=drone_path,
        start_inverted=start_inverted,
        randomize_reset=args.randomize_reset,
    )
    env = MinMaxObservationWrapper(env)
    env = NormalizeActionWrapper(env)

    # -----------------------
    # policy definition
    # -----------------------

    def policy(obs, key):
            return jnp.array([0.0, 0.0, 0.0, policy_eta])

    # -----------------------
    # run rollouts
    # -----------------------

    def get_rollouts(env, policy, num_rollouts, key):
        parallel_rollout = jax.vmap(rollout, in_axes=(None, 0, None))
        rollout_keys = jax.random.split(key, num_rollouts)
        transitions = parallel_rollout(env, rollout_keys, policy)
        return transitions

    transitions = get_rollouts(env, policy, args.num_drones, jax.random.key(3))
    goal_pos, goal_g_b = env.get_goal()

    plot_trajectories(transitions)

    if saving:
        save_trajectories(
            transitions,
            trial_name=trial_name,
            save_plots=args.save_plots,
            save_data=args.save_data,
        )

    metrics = eval_trajectories(
        traj=transitions, trial_name=trial_name, goal_g_b=goal_g_b, goal_pos=goal_pos, save_data=args.save_data
    )
    print_metrics(metrics, prefix="[TESTING_SCRIPT] ")

    # -----------------------
    # animation & still frames
    # -----------------------

    goal_R = jax.lax.select(
        start_inverted,
        jnp.eye(3),
        transform.Rotation.from_euler("xyz", jnp.array([0, jnp.pi, jnp.pi])).as_matrix(),
    )

    if args.animate or args.render_still_frames:
        animator = QuadrotorAnimator(
            goal_x=goal_pos,
            goal_R=goal_R,
            dpi=args.dpi,
            bitrate=args.bitrate,
            render_hz=args.render_hz,
            drone_scale=args.drone_scale,
            draw_trajectories=True,
        )

        if args.animate:
            animator.animate_trajectories(
                traj=transitions,
                trial_name=trial_name,
                title=animation_title,
                extension=args.extension,
                save_animation=args.save_animation
            )
            print("[EVAL_SCRIPT] Finished rendering the animation.")

        if args.render_still_frames:
            animator.plot_still_frames(
                traj=transitions,
                trial_name=trial_name,
                title=animation_title,
                drone_scale=args.drone_scale,
                num_poses=args.num_poses,
                flip_window=tuple(args.flip_window),
                colorbar=args.colorbar,
                save_still_frames=args.save_still_frames
            )
            print("[EVAL_SCRIPT] Finished rendering still frames.")

    # -----------------------
    # display
    # -----------------------

    if args.show_hud:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()