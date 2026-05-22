import jax
import jax.numpy as jnp
import jaxopt
from functools import partial


def get_segment_coeffs(d0, d1, T):
    """
    Maps 8 boundary constraints to 7th-order power basis coefficients natively.
    d0, d1: [pos, vel, acc, jerk] at start and end of segment.
    """

    c0, c1 = d0[0], d0[1]
    c2, c3 = d0[2] / 2.0, d0[3] / 6.0
    b0 = d1[0] - (c0 + c1 * T + c2 * T**2 + c3 * T**3)
    b1 = d1[1] - (c1 + 2 * c2 * T + 3 * c3 * T**2)
    b2 = d1[2] - (2 * c2 + 6 * c3 * T)
    b3 = d1[3] - (6 * c3)
    b = jnp.array([b0, b1, b2, b3])

    M = jnp.array(
        [
            [T**4, T**5, T**6, T**7],
            [4 * T**3, 5 * T**4, 6 * T**5, 7 * T**6],
            [12 * T**2, 20 * T**3, 30 * T**4, 42 * T**5],
            [24 * T, 60 * T**2, 120 * T**3, 210 * T**4],
        ]
    )

    c47 = jnp.linalg.solve(M, b)
    return jnp.concatenate([jnp.array([c0, c1, c2, c3]), c47])


def snap_cost_segment(c, T):
    """Analytically computes the integral of snap^2 over [0, T]."""
    # Snap is the 4th derivative. Coefficients: 24*c4, 120*c5*t, 360*c6*t^2, 840*c7*t^3
    snap_c = jnp.array([24.0 * c[4], 120.0 * c[5], 360.0 * c[6], 840.0 * c[7]])

    # Square the snap polynomial (equivalent to convolution of coefficients)
    snap_sq_c = jnp.convolve(snap_c, snap_c)

    # Integrate: increase power by 1, divide by new power
    powers = jnp.arange(1, 8)
    integ_c = snap_sq_c / powers

    # Evaluate at T (evaluation at 0 is just 0)
    T_powers = T**powers
    return jnp.sum(integ_c * T_powers)


def plan_1d_trajectory_jax(
    times,
    positions,
    start_v=0.0,
    ff_tuple=(),
    free_fall_accel=0.0,
    fix_jerk_at_ff=False,
    jerk_ff=0.0,
):
    n_points = len(positions)

    def objective(x, t, p):
        derivs = [jnp.array([p[0], start_v, 0.0, 0.0])]
        idx = 0
        for i in range(1, n_points - 1):
            vel = x[idx]
            idx += 1
            if i in ff_tuple:
                acc = free_fall_accel
                if fix_jerk_at_ff:
                    jerk = jerk_ff
                else:
                    jerk = x[idx]
                    idx += 1
            else:
                acc = x[idx]
                idx += 1
                jerk = x[idx]
                idx += 1
            derivs.append(jnp.array([p[i], vel, acc, jerk]))

        derivs.append(jnp.array([p[-1], 0.0, 0.0, 0.0]))

        cost = 0.0
        for i in range(n_points - 1):
            T = t[i + 1] - t[i]
            c = get_segment_coeffs(derivs[i], derivs[i + 1], T)
            cost += snap_cost_segment(c, T)

        return cost

    # Calculate number of variables
    n_vars = max(0, n_points - 2) * 3
    for i in range(1, n_points - 1):
        if i in ff_tuple:
            n_vars -= 1
            if fix_jerk_at_ff:
                n_vars -= 1

    initial_guess = jnp.zeros(n_vars)
    solver = jaxopt.BFGS(fun=objective)
    res = solver.run(initial_guess, t=times, p=positions)
    return res.params


def get_all_coeffs(
    optimized_x,
    times,
    positions,
    free_fall_indices,
    free_fall_accel,
    n_points,
    start_v=0.0,
    fix_jerk_at_ff=False,
    jerk_ff=0.0,
):
    derivs = [jnp.array([positions[0], start_v, 0.0, 0.0])]
    idx = 0
    for i in range(1, n_points - 1):
        vel = optimized_x[idx]
        idx += 1
        if i in free_fall_indices:
            acc = free_fall_accel
            if fix_jerk_at_ff:
                jerk = jerk_ff
            else:
                jerk = optimized_x[idx]
                idx += 1
        else:
            acc = optimized_x[idx]
            idx += 1
            jerk = optimized_x[idx]
            idx += 1
        derivs.append(jnp.array([positions[i], vel, acc, jerk]))

    derivs.append(jnp.array([positions[-1], 0.0, 0.0, 0.0]))

    coeffs = []
    for i in range(n_points - 1):
        T = times[i + 1] - times[i]
        coeffs.append(get_segment_coeffs(derivs[i], derivs[i + 1], T))

    return jnp.stack(coeffs)


def generate_3d_trajectory(
    times, waypoints, yaw_waypoints, s, start_v, free_fall_idx=()
):
    n_points = len(waypoints)
    waypoints = jnp.asarray(waypoints)

    # ax, ay, _ = 0.1 * (waypoints[free_fall_idx[0] + 1] - waypoints[free_fall_idx[0]])
    # ax, ay = 0.0, 0.0
    # jerk_cnt = False
    jerk_cnt = True

    # 1. Run JAXopt to find the optimal interior constraints (returns shape (N_vars,))
    opt_x = plan_1d_trajectory_jax(
        times, waypoints[:, 0], start_v[0], free_fall_idx, 0.0
    )
    opt_y = plan_1d_trajectory_jax(
        times, waypoints[:, 1], start_v[1], free_fall_idx, 0.0
    )
    # opt_z = plan_1d_trajectory_jax(times, waypoints[:, 2], free_fall_idx, -9.81)
    opt_z = plan_1d_trajectory_jax(
        times, waypoints[:, 2], start_v[2], free_fall_idx, -9.8, jerk_cnt, 0.0
    )
    opt_yaw = plan_1d_trajectory_jax(times, yaw_waypoints)

    # 2. Convert those raw variables into the full (N_segments, 8) coefficient matrices
    cx = get_all_coeffs(
        opt_x, times, waypoints[:, 0], free_fall_idx, 0.0, n_points, start_v[0]
    )
    cy = get_all_coeffs(
        opt_y, times, waypoints[:, 1], free_fall_idx, 0.0, n_points, start_v[1]
    )
    # cz = get_all_coeffs(opt_z, times, waypoints[:, 2], free_fall_idx, -9.81, n_points)
    cz = get_all_coeffs(
        opt_z,
        times,
        waypoints[:, 2],
        free_fall_idx,
        -9.8,
        n_points,
        start_v[2],
        jerk_cnt,
        0.0,
    )

    # Assuming yaw has no free-fall constraints
    cyaw = get_all_coeffs(opt_yaw, times, yaw_waypoints, (), 0.0, n_points)

    # These are what you pass into evaluate_polynomial_traj!
    return cx, cy, cz, cyaw, s


def evaluate_trajectory(coeffs, times, t_eval):
    """
    Evaluates the position across the piecewise polynomials.
    coeffs shape: (num_segments, 8)
    """
    # Find which segment each time point belongs to
    segment_indices = jnp.searchsorted(times, t_eval, side="right") - 1
    segment_indices = jnp.clip(segment_indices, 0, len(times) - 2)

    # Calculate local time within the segment
    t_local = t_eval - times[segment_indices]

    # Get the coefficients for the active segments
    c_active = coeffs[segment_indices]

    # Evaluate polynomial: sum(c_i * t^i) natively in JAX
    powers = jnp.arange(8)
    t_powers = jnp.power(t_local[:, None], powers[None, :])

    positions = jnp.sum(c_active * t_powers, axis=1)
    return positions


def minimum_snap(t, T, pos_final):
    # Minimum-snap trajectory
    tau = t / T

    pos = pos_final * (35 * tau**4 - 84 * tau**5 + 70 * tau**6 - 20 * tau**7)

    dpos = pos_final * ((140 * tau**3 - 420 * tau**4 + 420 * tau**5 - 140 * tau**6) / T)

    ddpos = pos_final * (
        (420 * tau**2 - 1680 * tau**3 + 2100 * tau**4 - 840 * tau**5) / T**2
    )

    dddpos = pos_final * (
        (840 * tau - 5040 * tau**2 + 8400 * tau**3 - 4200 * tau**4) / T**3
    )

    return pos, dpos, ddpos, dddpos


@jax.jit
def evaluate_polynomial_reference(times, cx, cy, cz, cyaw, eta_sequence, time):
    """
    Evaluate piecewise 7th-order polynomial trajectory at scalar time.

    Returns:
        p, v, acc, jrk, snp, yaw, dyaw, ddyaw, eta
    """
    idx = jnp.searchsorted(times, time, side="right") - 1
    idx = jnp.clip(idx, 0, times.shape[0] - 2)

    t_local = time - times[idx]

    def eval_single(c, d):
        coeff = c[idx]

        multiplier = jnp.ones(8)
        powers = jnp.arange(8)

        for _ in range(d):
            multiplier = multiplier * powers
            powers = jnp.maximum(powers - 1, 0)

        final_powers = jnp.maximum(jnp.arange(8) - d, 0)
        return jnp.sum(coeff * multiplier * (t_local**final_powers))

    p = jnp.array(
        [
            eval_single(cx, 0),
            eval_single(cy, 0),
            eval_single(cz, 0),
        ]
    )

    v = jnp.array(
        [
            eval_single(cx, 1),
            eval_single(cy, 1),
            eval_single(cz, 1),
        ]
    )

    acc = jnp.array(
        [
            eval_single(cx, 2),
            eval_single(cy, 2),
            eval_single(cz, 2),
        ]
    )

    jrk = jnp.array(
        [
            eval_single(cx, 3),
            eval_single(cy, 3),
            eval_single(cz, 3),
        ]
    )

    snp = jnp.array(
        [
            eval_single(cx, 4),
            eval_single(cy, 4),
            eval_single(cz, 4),
        ]
    )

    yaw = eval_single(cyaw, 0)
    dyaw = eval_single(cyaw, 1)
    ddyaw = eval_single(cyaw, 2)

    eta = jnp.asarray(eta_sequence, dtype=jnp.int32)[idx]

    return p, v, acc, jrk, snp, yaw, dyaw, ddyaw, eta


if __name__ == "__main__":
    times = jnp.array([0.0, 2.0, 4.0])
    waypoints = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 0.0, 0.0]])
    free_fall_idx = (1,)

    # Solve X, Y, Z independently
    opt_z = plan_1d_trajectory_jax(
        times,
        waypoints[:, 2],
        start_v=0.0,
        ff_tuple=free_fall_idx,
        free_fall_accel=-9.81,
    )
    # Get Coefficients
    coeffs_z = get_all_coeffs(
        opt_z, times, waypoints[:, 2], free_fall_idx, -9.81, len(waypoints)
    )

    # Evaluate
    t_eval = jnp.linspace(0, 4, 100)
    z_positions = evaluate_trajectory(coeffs_z, times, t_eval)

    print("Optimization Complete.")
    print(f"Z coefficients shape: {coeffs_z.shape}")
