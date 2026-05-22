import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
from orbax.checkpoint import PyTreeCheckpointer

from flightning import FLIGHTNING_PATH
from flightning.envs import QuadEnv
from flightning.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
from flightning.modules import ActorCriticPPO
import distrax


class MockDistribution:
    def __init__(self, loc, scale_diag):
        self.loc = loc

    def mean(self):
        return self.loc


distrax.MultivariateNormalDiag = MockDistribution


def build_env(drone_path: str, trajectory_path: str, dt: float = 0.02, delay: float = 0.001):
    env = QuadEnv(
        max_steps_in_episode=5 * int(1 / dt),
        dt=dt,
        delay=delay,
        roll_pitch_range=0.1,
        yaw_range=0.1,
        omega_std=0.1,
        margin=4.99999,
        drone_path=drone_path,
        trajectory_path=trajectory_path
    )
    env = MinMaxObservationWrapper(env)
    env = NormalizeActionWrapper(env)
    return env


def load_params(policy_path: str):
    policy_path = os.path.abspath(policy_path)
    ckptr = PyTreeCheckpointer()
    return ckptr.restore(policy_path, item=None)


def make_inference_fn(policy_net, params, env, bake_wrappers: bool):
    """
    bake_wrappers=True  : obs_raw (physical)  -> action_physical
    bake_wrappers=False : obs_norm ([-1,1])   -> action_norm ([-1,1])
    Always returns the distribution mean (deterministic, no sampling).
    """
    obs_min = env._obs_min
    obs_max = env._obs_max
    act_low = env._env.action_space.low
    act_high = env._env.action_space.high

    def infer_baked(obs_raw: jax.Array) -> jax.Array:
        obs_norm = 2.0 * (obs_raw - obs_min) / (obs_max - obs_min) - 1.0
        action_norm = policy_net.apply(params, obs_norm).pi.mean()
        action_phys = (action_norm + 1.0) / 2.0 * (act_high - act_low) + act_low
        return action_phys

    def infer_net_only(obs_norm: jax.Array) -> jax.Array:
        return policy_net.apply(params, obs_norm).pi.mean()

    return infer_baked if bake_wrappers else infer_net_only


def export_to_onnx(infer_fn, obs_dim: int, output_path: str, bake_wrappers: bool):
    """
    Export using jax2onnx with batched I/O:
        input  shape: (1, obs_dim)
        output shape: (1, action_dim)
    """
    try:
        from jax2onnx import to_onnx
    except ImportError as exc:
        raise ImportError("Run: pip install jax2onnx") from exc

    obs_name = "obs_raw" if bake_wrappers else "obs_norm"
    act_name = "action_physical" if bake_wrappers else "action_norm"

    # NOTE: Using shape (1, obs_dim) to satisfy ONNX Gemm rank-2 requirement
    dummy = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    model = to_onnx(infer_fn, [dummy])

    # Safely rename the input across the entire graph
    old_input_name = model.graph.input[0].name
    model.graph.input[0].name = obs_name
    for node in model.graph.node:
        for i, node_in in enumerate(node.input):
            if node_in == old_input_name:
                node.input[i] = obs_name

    # Safely rename the output across the entire graph
    old_output_name = model.graph.output[0].name
    model.graph.output[0].name = act_name
    for node in model.graph.node:
        for i, node_out in enumerate(node.output):
            if node_out == old_output_name:
                node.output[i] = act_name

    import onnx

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"[EXPORTING_SCRIPT] Saved ONNX model to: {output_path}")


def verify(infer_fn, output_path: str, obs_dim: int, n_samples: int = 64):
    """
    Pass random batched observation vectors through both the JAX function and the
    ONNX runtime and check they agree within float32 tolerance.
    """

    import onnxruntime as ort

    sess = ort.InferenceSession(output_path)
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(42)
    obs_batch = rng.uniform(-1.0, 1.0, size=(n_samples, obs_dim)).astype(np.float32)

    max_err = 0.0
    worst = {}

    for i in range(n_samples):
        # NOTE: Slice to keep shape as (1, obs_dim)
        obs_np = obs_batch[i : i + 1]

        jax_out = np.asarray(infer_fn(jnp.asarray(obs_np)))
        ort_out = sess.run(None, {input_name: obs_np})[0]  # (1, action_dim)
        err = float(np.max(np.abs(jax_out - ort_out)))

        if err > max_err:
            max_err = err
            worst = {"idx": i, "jax": jax_out, "ort": ort_out, "obs": obs_np}

    tol = 1e-4
    status = "PASS" if max_err < tol else "FAIL"
    print(f"[EXPORTING_SCRIPT] Validation status: {status}")

    if max_err >= tol:
        raise RuntimeError(
            "ONNX verification failed — outputs diverge beyond tolerance."
        )


def parse_args():
    p = argparse.ArgumentParser(description="Export a flightning policy to ONNX.")
    p.add_argument("--policy-path", required=True, help="path to the flax params")
    p.add_argument("--drone-path", default=None, help="path to the drone for validation")
    p.add_argument("--output", required=True, help="output ONNX model name")
    p.add_argument("--dt", type=float, default=0.02, help="dt for evaluation")
    p.add_argument("--delay", type=float, default=0.001, help="delay for evaluation")
    p.add_argument("--network-only", action="store_true", help="flag to not include wrappers")
    p.add_argument("--verify", action="store_true", help="flag to verify ONNX policy export")
    return p.parse_args()


def main():
    args = parse_args()

    drone_path = args.drone_path or (
        FLIGHTNING_PATH + "/objects/quadrotor_files/eris03_eval.yaml"
    )
    trajectory_path = FLIGHTNING_PATH + "/trajectories/config/trajectories.yaml"

    bake_wrappers = not args.network_only

    env = build_env(drone_path, trajectory_path, dt=args.dt, delay=args.delay)

    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    policy_net = ActorCriticPPO(
        [obs_dim, 512, 512, action_dim],
        initial_log_std=jnp.log(0.25),
    )

    params = load_params(args.policy_path)
    print(f"[EXPORTING_SCRIPT] Loaded checkpoint: {args.policy_path}")

    infer_fn = make_inference_fn(policy_net, params, env, bake_wrappers)

    dummy = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    out = infer_fn(dummy)


    export_to_onnx(infer_fn, obs_dim, args.output, bake_wrappers)

    if args.verify:
        verify(infer_fn, args.output, obs_dim)

    print("[EXPORTING_SCRIPT] Deployment notes")
    print("=" * 60)
    print(f"  Input  shape : (1, {obs_dim})")
    print(f"  Output shape : (1, {action_dim})")
    print("=" * 60)


if __name__ == "__main__":
    main()
