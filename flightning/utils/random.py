import chex
import jax.random as jrandom
import jax.numpy as jnp
import jax.scipy.spatial.transform as transform


def key_generator(seed):
    """
    Generator for random keys. Use for debugging and testing only!
    >>> key_gen = key_generator(0)
    >>> key1 = next(key_gen)
    >>> key2 = next(key_gen)
    """
    key = jrandom.key(seed)
    while True:
        key, subkey = jrandom.split(key)
        yield subkey


def random_rotation_matrix(key: jnp.ndarray) -> jnp.ndarray:
    """Generates a random rotation matrix."""
    random_vec = jrandom.normal(key, (3,))
    return transform.Rotation.from_rotvec(random_vec).as_matrix()


def random_rotation(
    key: chex.PRNGKey, yaw_range: float, pitch_range: float, roll_range: float
) -> transform.Rotation:
    key_yaw, key_pitch, key_roll = jrandom.split(key, 3)
    yaw = jrandom.uniform(key_yaw, minval=-yaw_range, maxval=yaw_range)
    pitch = jrandom.uniform(key_pitch, minval=-pitch_range, maxval=roll_range)
    roll = jrandom.uniform(key_roll, minval=-roll_range, maxval=roll_range)
    # convert to rotation matrix (assuming extrinsic rotations)
    rotation = transform.Rotation.from_euler("zyx", jnp.array([yaw, pitch, roll]))
    return rotation


def random_rotation_vector_limited(
    key: chex.PRNGKey,
    angle_limit: float,
) -> transform.Rotation:
    key_dir, key_rad = jrandom.split(key)

    v = jrandom.normal(key_dir, (3,))
    v = v / (jnp.linalg.norm(v) + 1e-8)

    u = jrandom.uniform(key_rad)
    r = angle_limit * u ** (1.0 / 3.0)  # uniform in 3D ball

    rotvec = r * v
    return transform.Rotation.from_rotvec(rotvec)
