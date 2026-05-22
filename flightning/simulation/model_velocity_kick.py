from typing import NamedTuple

import jax
import jax.numpy as jnp


class VelocityKickParams(NamedTuple):
    kick_magnitude: float
    kick_frequency: float


def simulate_velocity_kick(dt, key, params: VelocityKickParams) -> jnp.ndarray:
    """
    Velocity kick as instantaneous delta-v.
    Approximate a Poisson process by applying a Bernoulli with p = kick_frequency * dt.
    """
    dt = jnp.asarray(dt, dtype=jnp.float32)
    p = jnp.clip(params.kick_frequency * dt, 0.0, 1.0)

    key_event, key_dir, key_mag = jax.random.split(key, 3)
    event = (jax.random.uniform(key_event, ()) < p).astype(jnp.float32)

    dir_raw = jax.random.normal(key_dir, (3,))
    dir_unit = dir_raw / (jnp.linalg.norm(dir_raw) + 1e-8)

    # optional magnitude randomness (lognormal-ish)
    mag = params.kick_magnitude * (0.5 + jax.random.uniform(key_mag, ()))
    dv = event * mag * dir_unit
    return dv
