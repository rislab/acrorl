import os
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import jax.scipy.spatial.transform as transform
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer

from flightning import FLIGHTNING_PATH
from flightning.algos import ppo
from flightning.envs import QuadEnv, rollout
from flightning.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
from flightning.modules import ActorCriticPPO
from flightning.utils.printing import print_metrics
from flightning.utils.plotting import plot_trajectories, save_trajectories
from flightning.utils.evaluation import eval_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="Train a learned inversion policy.")

    parser.add_argument("--inversion_type", type=str, choices=["nti", "itn"], required=True,
                        help="inversion type: 'nti' (nominal to inverted) or 'itn' (inverted to nominal).")
    parser.add_argument("--trial_name", type=str, default=None,
                        help="name for saving plots, data, and policy; defaults to inversion_type + '_learned'")
    parser.add_argument("--num_drones", type=int, default=10,
                        help="number of parallel rollouts before training (default: 10)")
    parser.add_argument("--num_drones_eval", type=int, default=10,
                        help="number of parallel rollouts during eval (default: 10)")
    parser.add_argument("--duration", type=int, default=3,
                        help="training episode duration in seconds (default: 3)")
    parser.add_argument("--duration_eval", type=int, default=5,
                        help="eval episode duration in seconds (default: 5)")
    parser.add_argument("--drone_name", type=str, default="eris03",
                        help="base drone yaml filename without suffix or .yaml (default: eris03)")

    # PPO
    parser.add_argument("--num_epochs", type=int, default=750,
                        help="number of PPO training epochs (default: 750)")
    parser.add_argument("--num_envs", type=int, default=2048,
                        help="number of parallel envs for PPO (default: 2048)")
    parser.add_argument("--num_minibatches", type=int, default=20,
                        help="number of PPO minibatches (default: 20)")
    parser.add_argument("--entropy_coefficient", type=float, default=0.01,
                        help="PPO entropy coefficient (default: 0.01)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="PPO discount factor (default: 0.99)")
    parser.add_argument("--update_epochs", type=int, default=4,
                        help="PPO update epochs per iteration (default: 4)")

    # Saving
    parser.add_argument("--save_plots", action="store_true",
                        help="save trajectory plots as PNGs to disk")
    parser.add_argument("--save_data", action="store_true",
                        help="save trajectory data as CSVs to disk")
    parser.add_argument("--show_hud", action="store_true",
                        help="show all figures interactively after rendering")

    return parser.parse_args()


# -----------------------
# hardcoded constants
# -----------------------

seed = 0
dt = 0.02
delay = 0.006

# training env
margin = 2.5
velocity_std = 0.1
roll_pitch_range = jnp.pi / 4
yaw_range = jnp.pi
omega_std = 0.1
randomize_reset = True
trajectory_profile = "constant_reference"
trajectory_path = FLIGHTNING_PATH + "/trajectories/config/trajectories.yaml"

# eval env
margin_eval = 4.9
velocity_std_eval = 0.1
roll_pitch_range_eval = 0.1
yaw_range_eval = jnp.pi
omega_std_eval = 0.1
randomize_reset_eval = True

# network
hidden_layer_dims = [512, 512]
initial_log_std = jnp.log(0.25)
adam_learning_rate = 3e-4
action_history = 3


def main():
    args = parse_args()

    start_inverted = args.inversion_type == "itn"
    trial_name = args.trial_name if args.trial_name is not None else f"{args.inversion_type}_learned"
    saving = args.save_plots or args.save_data

    drone_path      = f"{FLIGHTNING_PATH}/objects/quadrotor_files/{args.drone_name}_train.yaml"
    drone_path_eval = f"{FLIGHTNING_PATH}/objects/quadrotor_files/{args.drone_name}_eval_oca.yaml"

    # -----------------------
    # env definition
    # -----------------------

    key = jax.random.key(seed)
    key_init, key_ppo = jax.random.split(key, 2)

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
        margin=margin,
        drone_path=drone_path,
        start_inverted=start_inverted,
        randomize_reset=randomize_reset,
        action_history=action_history,
    )
    env = MinMaxObservationWrapper(env)
    env = NormalizeActionWrapper(env)

    # -----------------------
    # policy definition
    # -----------------------

    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    policy_net = ActorCriticPPO(
        [obs_dim, hidden_layer_dims[0], hidden_layer_dims[1], action_dim],
        initial_log_std=initial_log_std,
    )
    policy_params = policy_net.initialize(key_init)
    tx = optax.adam(adam_learning_rate)
    train_state = TrainState.create(
        apply_fn=policy_net.apply,
        params=policy_params,
        tx=tx,
    )

    def policy(obs, key):
        pi = train_state.apply_fn(train_state.params, obs).pi
        return pi.sample(seed=key)

    # -----------------------
    # initial rollouts
    # -----------------------

    def get_rollouts(env, policy, num_rollouts, key):
        parallel_rollout = jax.vmap(rollout, in_axes=(None, 0, None))
        rollout_keys = jax.random.split(key, num_rollouts)
        return parallel_rollout(env, rollout_keys, policy)

    transitions_init = get_rollouts(env, policy, args.num_drones, jax.random.key(seed))
    plot_trajectories(transitions_init)
    if saving:
        save_trajectories(
            transitions_init,
            trial_name=f"{trial_name}_init",
            save_plots=args.save_plots,
            save_data=args.save_data,
        )
    print("[TRAINING_SCRIPT] Plotted initial rollouts.")

    # -----------------------
    # train
    # -----------------------

    print("[TRAINING_SCRIPT] Started training.")
    time_start = time.time()
    res_dict = ppo.train(
        env,
        train_state,
        num_epochs=args.num_epochs,
        num_steps_per_epoch=env.max_steps_in_episode,
        num_envs=args.num_envs,
        key=key_ppo,
        config=ppo.Config(
            num_minibatches=args.num_minibatches,
            ent_coef=args.entropy_coefficient,
            gamma=args.gamma,
            update_epochs=args.update_epochs,
        ),
    )
    time_train = time.time() - time_start
    print(f"[TRAINING_SCRIPT] Finished training in {time_train:.2f} [s].")

    # -----------------------
    # training metrics
    # -----------------------

    episode_returns = res_dict["metrics"]["returned_episode_returns"]
    returned_episode = res_dict["metrics"]["returned_episode"]
    returns = episode_returns * returned_episode
    num_returned_episodes = returned_episode.sum(axis=(1, 2))
    mean_returns = returns.sum(axis=(1, 2)) / jnp.maximum(num_returned_episodes, 1)
    mean_returns_np = np.array(mean_returns)

    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    np.savetxt(
        f"data/{trial_name}_rewards.csv",
        mean_returns_np,
        delimiter=",",
        header="mean_return",
        comments="",
    )
    plt.figure()
    plt.plot(jnp.array(mean_returns))
    plt.title(f"Final Return: {mean_returns[-1]:.2f}, Training Time: {time_train:.2f} [s]")
    plt.xlabel("Iteration")
    plt.ylabel("Return")
    plt.savefig(f"plots/{trial_name}_rewards.png")
    print("[TRAINING_SCRIPT] Plotted rewards.")

    # -----------------------
    # trained policy
    # -----------------------

    new_train_state = res_dict["runner_state"].train_state

    def policy_trained(obs, key):
        pi = new_train_state.apply_fn(new_train_state.params, obs).pi
        return pi.mean()

    # -----------------------
    # eval env definition
    # -----------------------

    env_eval = QuadEnv(
        max_steps_in_episode=args.duration_eval * int(1 / dt),
        dt=dt,
        trajectory_profile=trajectory_profile,
        trajectory_path=trajectory_path,
        delay=delay,
        roll_pitch_range=roll_pitch_range_eval,
        yaw_range=yaw_range_eval,
        omega_std=omega_std_eval,
        velocity_std=velocity_std_eval,
        margin=margin_eval,
        drone_path=drone_path_eval,
        start_inverted=start_inverted,
        randomize_reset=randomize_reset_eval,
        action_history=action_history,
    )
    env_eval = MinMaxObservationWrapper(env_eval)
    env_eval = NormalizeActionWrapper(env_eval)

    # -----------------------
    # eval rollouts
    # -----------------------

    transitions_eval = get_rollouts(env_eval, policy_trained, args.num_drones_eval, jax.random.key(seed))
    plot_trajectories(transitions_eval)
    if saving:
        save_trajectories(
            transitions_eval,
            trial_name=f"{trial_name}_end",
            save_plots=args.save_plots,
            save_data=args.save_data,
        )
    goal_pos, goal_g_b = env_eval.get_goal()
    metrics = eval_trajectories(
        traj=transitions_eval, trial_name=trial_name if saving else None,
        goal_g_b=goal_g_b, goal_pos=goal_pos
    )
    print_metrics(metrics, prefix="[TRAINING_SCRIPT] ")

    # -----------------------
    # display
    # -----------------------

    if args.show_hud:
        plt.show()
    else:
        plt.close("all")

    # -----------------------
    # policy saving
    # -----------------------

    while True:
        ans = input("[TRAINING_SCRIPT] Would you like to save the policy params? (y/n): ").strip().lower()
        if ans == "y":
            name = input("[TRAINING_SCRIPT] What name would you like to give it? ")
            path = f"{FLIGHTNING_PATH}/../policies/{name}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ckptr = PyTreeCheckpointer()
            ckptr.save(path, new_train_state.params)
            print(f"[TRAINING_SCRIPT] Policy saved to: {path}")
            break
        elif ans == "n":
            print("[TRAINING_SCRIPT] Policy not saved.")
            break
        else:
            print("[TRAINING_SCRIPT] Invalid input. Please enter 'y' or 'n'.")


if __name__ == "__main__":
    main()