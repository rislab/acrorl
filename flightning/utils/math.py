import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Any, Dict, List, Tuple


def skew(v: jnp.ndarray) -> jnp.ndarray:
    """Returns the skew symmetric matrix of a 3D vector."""
    return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def vec(M: jnp.ndarray) -> jnp.ndarray:
    """Converts the matrix into a vector."""
    return M.flatten()


def vee(S):
    """Returns the vee"""
    return jnp.array([S[2, 1], S[0, 2], S[1, 0]])


def devec(v: jnp.ndarray) -> jnp.ndarray:
    """Converts the vector into a 3x3 matrix."""
    return v.reshape(3, 3)


def rotation_matrix_from_vector(v):
    eps = 1e-5
    K = skew(v)
    theta = jnp.linalg.norm(v) + eps
    sin_term = jnp.sin(theta) / theta
    cos_term = (1 - jnp.cos(theta)) / (theta**2)
    I = jnp.eye(3)
    K2 = K @ K
    R = I + sin_term * K + cos_term * K2
    return R


def euler_from_rotation_matrix(R):
    """Returns ZYX euler angles from a rotation matrix."""
    rot = Rotation.from_matrix(R)
    return rot.as_euler(seq="zyx", degrees=False)


def euler_rates_from_body_rates(euler, omega, eps=1e-6):
    """Returns euler rates from a euler angles and body rates."""
    theta = euler[1]  # pitch
    phi = euler[2]  # roll

    c = jnp.cos
    s = jnp.sin
    t = jnp.tan
    ct = c(theta) + eps

    A = jnp.array(
        [
            [1, s(phi) * t(theta), c(phi) * t(theta)],
            [0, c(phi), -s(phi)],
            [0, s(phi) / ct, c(phi) / ct],
        ]
    )
    return A @ omega


def special_sign(v: jnp.ndarray) -> jnp.ndarray:
    """Returns the sign of the vector, with 0 mapped to 1."""
    return jnp.sign(v) + (v == 0)


def l1(x):
    return jnp.sum(jnp.abs(x))


def huber_l1(x, delta=0.1):
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    linear = abs_x - quadratic
    return jnp.sum(0.5 * quadratic**2 + delta * linear)


def smooth_l1(x):
    delta = 1.0
    abs_errors = jnp.linalg.norm(x) + 1e-6  # epsilon OUTSIDE norm
    quadratic = jnp.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return 0.5 * quadratic**2 + delta * linear


def rot_from_quat(q: jnp.ndarray) -> Rotation:
    """
    Convert quaternion to rotation object (using another convention than
    scipy.spatial.transform.Rotation.from_quat)
    :param q: quaternion as (cos(theta/2), sin(theta/2) * axis)
    :return: rotation object
    """
    # permute the quaternion to match the convention of
    # scipy.spatial.transform.Rotation
    q = jnp.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(q)


def rot_to_quat(rot: Rotation) -> jnp.array:
    """
    Convert rotation object to quaternion
    :param rot: rotation object
    :return: rotation quaternion as (cos(theta/2), sin(theta/2) * axis)
    """
    q = rot.as_quat()
    return jnp.array([q[3], q[0], q[1], q[2]])


def normalize(a, a_min, a_max):
    """
    Maps input a from [a_min, a_max] to [-1, 1]
    """
    return 2 * (a - a_min) / (a_max - a_min) - 1


def proj_gravity(R) -> jnp.array:
    g_w = jnp.array([0.0, 0.0, -1.0])
    return R.T @ g_w


def rbf_from_sqerr(err_sq: jnp.ndarray, std: float) -> jnp.ndarray:
    """exp(-||e||^2 / std^2), numerically safe and in [0,1]."""
    s2 = (std**2) + 1e-12
    return jnp.exp(-err_sq / s2)


def rbf_vec(x: jnp.ndarray, std: float, axis=-1) -> jnp.ndarray:
    """exp(-||x||^2 / std^2) over the last axis by default."""
    return rbf_from_sqerr(jnp.sum(x * x, axis=axis), std)


def l2_penalty(vec: jnp.ndarray) -> jnp.ndarray:
    """Quadratic penalty"""
    return jnp.sum(vec**2)


def _l2_normalize(x: jnp.array, eps: float = 1e-12) -> jnp.array:
    return x / (jnp.linalg.norm(x) + eps)


def sigmoid(beta, xi):
    """sigmoid function"""
    return 1.0 / (1.0 + jnp.exp(-beta / xi))


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_normalize(q):
    return q / jnp.linalg.norm(q, keepdims=True)


def quat_conj(q):
    q = -q
    q = q.at[0].set(-q[0])
    return q


def quat_inv(q):
    return quat_conj(q) / (jnp.linalg.norm(q, keepdims=True)) ** 2


def quat_to_rot(q):
    q = quat_normalize(q)
    w, x, y, z = q
    R = jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R


def quat_error_shaped(qd, q, eps=1e-6):
    qd_inv = quat_inv(qd)
    w = qd_inv[0] * q[0] - jnp.dot(qd_inv[1:], q[1:])
    v = qd_inv[0] * q[1:] + q[0] * qd_inv[1:] + jnp.cross(qd_inv[1:], q[1:])

    # shortest rotation
    sign = jnp.sign(w + 1e-8)
    w = w * sign
    v = v * sign

    # angle
    v_norm = jnp.linalg.norm(v)
    theta = 2.0 * jnp.arctan2(v_norm, w)

    # shaped error
    e = theta * v / (v_norm + eps)
    return e, jnp.array([w, v[0], v[1], v[2]])


def skew_symmetric_error(Rd, R):
    Re = Rd.T @ R
    return 0.5 * vee(Re - Re.T)


def softplus_stable(x, beta):
    return jax.nn.softplus(beta * x) / beta


def smooth_clip(x, lo, hi, beta):
    return lo + softplus_stable(x - lo, beta) - softplus_stable(x - hi, beta)


def sigmoid_Jr(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def Jr_inv(phi, eps=1e-4, eps_pi=1e-4):
    """
    Inverse right Jacobian on SO(3): Jr^{-1}(phi), phi in R^3.
    Stable near 0 (series) and near pi (linear approx).
    """
    I = jnp.eye(3)
    theta = jnp.linalg.norm(phi)
    W = skew(phi)
    W2 = W @ W
    pi = jnp.pi

    # small-angle series: a = 1/12 + theta^2/720
    a_small = 1.0 / 12.0 + (theta * theta) / 720.0

    # regular: a = 1/theta^2 - (1+cosθ)/(2 θ sinθ)
    sin_t = jnp.sin(theta)
    cos_t = jnp.cos(theta)
    a_reg = (1.0 / (theta * theta)) - (1.0 + cos_t) / (2.0 * theta * sin_t)

    # near-pi: a ≈ 1/pi^2 + (pi^2 - 8)(theta-pi)/(4 pi^3)  (paper Eq 18c)
    a_pi = (1.0 / (pi * pi)) + ((pi * pi - 8.0) * (theta - pi)) / (4.0 * pi**3)

    use_small = theta < eps
    use_pi = theta > (pi - eps_pi)
    a = jnp.where(use_small, a_small, jnp.where(use_pi, a_pi, a_reg))

    return I + 0.5 * W + a * W2
