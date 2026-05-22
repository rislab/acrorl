import jax
import jax.numpy as jnp
from typing import NamedTuple

from flightning.utils.math import sigmoid


class ActuatorModelParams(NamedTuple):
    motor_omega_min: float
    motor_omega_max: float
    motor_omega_0: jnp.ndarray
    motor_omega_dead: float
    motor_tau_pos: float
    motor_tau_neg: float
    motor_tau_dr: bool
    motor_inertia: float
    motor_directions: jnp.ndarray
    full_thrust_model: bool
    cT_pos: jnp.ndarray
    cT_neg: jnp.ndarray
    cT_dr: bool
    motor_xi: float
    Q_u: jnp.ndarray
    use_opt_alloc: bool
    

class ActuatorModel:
    def __init__(self, quadrotor, params: ActuatorModelParams):
        self.quadrotor = quadrotor
        self.params = params

        # thrust coefficients
        if jnp.asarray(params.cT_pos).ndim > 0 and jnp.asarray(params.cT_pos).shape[0] == 3:
            self.cT_simple_pos = jnp.asarray(params.cT_pos[0])
            self.cT_simple_neg = jnp.asarray(params.cT_neg[0])
        else:
            self.cT_simple_pos = jnp.asarray(params.cT_pos)
            self.cT_simple_neg = jnp.asarray(params.cT_neg)

        self.cT_full_pos = jnp.asarray(params.cT_pos) 
        self.cT_full_neg = jnp.asarray(params.cT_neg)

        # thrust limits
        self.T_min = self.get_T_deterministic(jnp.full(4, self.quadrotor._motor_omega_min))
        self.T_max = self.get_T_deterministic(jnp.full(4, self.quadrotor._motor_omega_max))

        # control allocation weighting matrix
        self.Q_u = jnp.diag(self.params.Q_u)

    def step(self, key, motor_omega, motor_omega_d, dt):
        """ step actuator dynamics with domain randomization """

        # motor_omega_0 domain randomization
        motor_omega_0 = jax.random.uniform(key,
            minval=self.params.motor_omega_0[0],
            maxval=self.params.motor_omega_0[1],
        )

        # time constant switch and domain randomization
        motor_tau = jnp.where(
            motor_omega >= motor_omega_0,
            self.params.motor_tau_pos,
            self.params.motor_tau_neg,
        )
        motor_tau = jax.lax.cond(
            self.params.motor_tau_dr,
            lambda _: (jax.random.uniform(key, motor_tau.shape, minval=0.75*motor_tau, maxval=1.25*motor_tau)),
            lambda _: (motor_tau),
            operand=None
        )

        # switching regions logic
        switch_region = jnp.logical_or(
            jnp.logical_and(
                motor_omega > (motor_omega_0 + self.params.motor_omega_dead),
                motor_omega_d < motor_omega_0,
            ),
            jnp.logical_and(
                motor_omega < (motor_omega_0 - self.params.motor_omega_dead),
                motor_omega_d > motor_omega_0,
            ),
        )
        motor_target = jnp.where(
            switch_region,
            motor_omega_0,
            jnp.clip(motor_omega_d, self.params.motor_omega_min, self.params.motor_omega_max),
        )

        # step dynamics
        motor_omega_new = (motor_omega - motor_target) * jnp.exp(-dt / motor_tau) + motor_target
        motor_omega_new = jnp.clip(
            motor_omega_new, self.params.motor_omega_min, self.params.motor_omega_max
        )

        dmotor_omega = (motor_omega_new - motor_omega) / dt
        motor_inertia_torque = jnp.array(
            [0.0, 0.0, (dmotor_omega * -self.params.motor_directions).sum() * self.params.motor_inertia]
        )
        return motor_inertia_torque, motor_omega_new
    

    def allocate_control(self, u_des, M, T_prev, motor_omega):
        """ control allocation from u_des to desired motor rates """
        if self.params.use_opt_alloc:
            return self._optimal_control_allocation(u_des, M, T_prev, motor_omega)
        else:
            return self._simple_control_allocation(u_des, M, motor_omega)

    def _optimal_control_allocation(self, u_des, M, T_prev, motor_omega):
        """ projected gradient gescent in thrust space to get optimal control allocation """
        
        # optimization params
        reg = 1e-3
        n_iter = 12
        
        # build the hessian (constant wrt T)
        A = self.Q_u @ M # (4,4)
        H = A.T @ A + reg * jnp.eye(4, dtype=M.dtype)
        f = -(A.T @ (self.Q_u @ u_des) + reg * T_prev)

        # unconstrained optimum: T* = -H^{-1} f
        # gradient is:  H T + f
        # step size from spectral radius of H
        L = jnp.linalg.eigvalsh(H).max()
        step = 1.0 / (L + 1e-8)

        # warm step
        T0 = jnp.clip(T_prev, self.T_min, self.T_max)

        def pgd_step(T, _):
            grad = H @ T + f
            T_new = jnp.clip(T - step * grad, self.T_min, self.T_max)
            return T_new, None

        T_cmd, _ = jax.lax.scan(pgd_step, T0, xs=None, length=n_iter)

        motor_omega_d = self.get_motor_omega_d(T_cmd, motor_omega)
        return motor_omega_d, T_cmd
    
    def _simple_control_allocation(self, u_des, M, motor_omega):
        """ baseline control allocation"""
        T_cmd = jnp.linalg.pinv(M) @ u_des
        T_cmd = jnp.clip(T_cmd, self.T_min, self.T_max)
        motor_omega_d = self.get_motor_omega_d(T_cmd, motor_omega)
        return motor_omega_d, T_cmd

    def _simple_model_T(self, operand):
        """ simple thrust model:  T = cT2(w)*w|w| """
        key, motor_omega = operand
        s = sigmoid(motor_omega, self.params.motor_xi)
        cT = s * self.cT_simple_pos + (1.0 - s) * self.cT_simple_neg
        if self.params.cT_dr:
            cT = jax.random.uniform(key, cT.shape, minval=0.9 * cT, maxval=1.1 * cT)
        T = cT * motor_omega * jnp.abs(motor_omega)
        return T

    def _full_model_T(self, operand):
        """ full thrust model:  T = cT2(w)*w|w| + cT1(w)*w + cT0(w) """
        key, motor_omega = operand
        s = sigmoid(motor_omega, self.params.motor_xi)
        cT = s[:, None] * self.cT_full_pos + (1.0 - s[:, None]) * self.cT_full_neg
        if self.params.cT_dr:
            cT = jax.random.uniform(key, cT.shape, minval=0.9 * cT, maxval=1.1 * cT)

        def _motor_thrust(c_i, o_i):
            o_vec = jnp.array([o_i * jnp.abs(o_i), o_i, 1.0])
            return jnp.dot(c_i, o_vec)

        T = jax.vmap(_motor_thrust)(cT, motor_omega)
        return T

    def get_T(self, key, motor_omega):
        if self.params.full_thrust_model:
            return self._full_model_T((key, motor_omega))
        else:
            return self._simple_model_T((key, motor_omega))

    def get_T_deterministic(self, motor_omega):
        """ deterministic thrust model """
        if self.params.full_thrust_model:
            s = sigmoid(motor_omega, self.params.motor_xi)
            cT = s[:, None] * self.cT_full_pos + (1.0 - s[:, None]) * self.cT_full_neg

            def _motor_thrust(c_i, o_i):
                o_vec = jnp.array([o_i * jnp.abs(o_i), o_i, 1.0])
                return jnp.dot(c_i, o_vec)

            return jax.vmap(_motor_thrust)(cT, motor_omega)
        else:
            s = sigmoid(motor_omega, self.params.motor_xi)
            cT = s * self.cT_simple_pos + (1.0 - s) * self.cT_simple_neg
            return cT * motor_omega * jnp.abs(motor_omega)
        
    def get_motor_omega_d(self, T_d, motor_omega):
        if self.params.full_thrust_model:
            return self._full_model_motor_omega_d((T_d, motor_omega))
        else:
            return self._simple_model_motor_omega_d((T_d, motor_omega))

    def _simple_model_motor_omega_d(self, operand):
        T_d, motor_omega = operand
        s = sigmoid(motor_omega, self.params.motor_xi)
        cT = s * self.cT_simple_pos + (1.0 - s) * self.cT_simple_neg
        return jnp.where(T_d >= 0.0, jnp.sqrt(T_d / cT), -jnp.sqrt(-T_d / cT))

    def _full_model_motor_omega_d(self, operand):
        """ Quadratic formula """
        T_d, motor_omega = operand
        s = sigmoid(motor_omega, self.params.motor_xi)
        cT = s[:, None] * self.cT_full_pos + (1.0 - s[:, None]) * self.cT_full_neg
        cT2 = cT[:, 0]
        cT1 = cT[:, 1]
        cT0 = cT[:, 2]
        T_abs = jnp.abs(T_d)
        discriminant = jnp.maximum(cT1 ** 2 - 4 * cT2 * (jnp.abs(cT0) - T_abs), 0.0)
        omega = (-cT1 + jnp.sqrt(discriminant)) / (2 * cT2)
        return jnp.where(T_d >= 0.0, omega, -omega)

if __name__ == "__main__":
    """CONTROL ALLOCATION ABLATION"""
    import time
    from jaxopt import BoxCDQP

    class _QuadrotorImitator:
        def __init__(self, omega_min, omega_max):
            self._motor_omega_min = omega_min
            self._motor_omega_max = omega_max

    OMEGA_MAX = 2000.0
    OMEGA_MIN = -2000.0
    CT_POS = jnp.array([1e-6, 1e-3, 1])
    CT_NEG = CT_POS/2
    MOTOR_XI  = 5.0
    MASS = 1.0
    G = 9.81
    REG = 1e-3

    actuator_params = ActuatorModelParams(
        motor_omega_min  = OMEGA_MIN,
        motor_omega_max  = OMEGA_MAX,
        motor_omega_0    = jnp.array([OMEGA_MIN, OMEGA_MAX]),   # DR range (unused here)
        motor_omega_dead = 1.0,
        motor_tau_pos    = 0.1,
        motor_tau_neg    = 0.1,
        motor_tau_dr     = False,
        motor_inertia    = 1.0e-6,
        motor_directions = jnp.array([1.0, 1.0, -1.0, -1.0]),
        full_thrust_model= False,
        cT_pos           = CT_POS,
        cT_neg           = CT_NEG,
        cT_dr            = False,
        motor_xi         = MOTOR_XI,
        Q_u              = jnp.array([1.0, 15.0, 15.0, 6.0]),
        use_opt_alloc    = True,
    )

    quadrotor_imitation = _QuadrotorImitator(omega_min=OMEGA_MIN, omega_max=OMEGA_MAX)
    actuator_model = ActuatorModel(quadrotor_imitation, actuator_params)
    T_MAX = float(actuator_model.T_max[0])
    T_MIN = float(actuator_model.T_min[0])
    T_HOVER = MASS * G / 4.0
    W_HOVER = actuator_model.get_motor_omega_d(jnp.full(4, T_HOVER), jnp.full(4, OMEGA_MAX))
    moment_scale = 0.001
    motor_directions = jnp.array([1.0, 1.0, -1.0, -1.0])
    ys = jnp.array([-0.1,  0.1, 0.1, -0.1])
    xs = jnp.array([ 0.1, -0.1, 0.1, -0.1])
    mixer_matrix = jnp.array(
        [jnp.ones(4), ys, -xs, moment_scale * -motor_directions], dtype=jnp.float32
    )
    Q_u = jnp.diag(jnp.array([100.0, 1500.0, 1500.0, 600.0], dtype=jnp.float32))
    T_prev = jnp.full(4, T_HOVER)

    print("[ACTUATOR_MODEL] Control Allocation Ablation Script")
    print("-" * 50)

    box_qp = BoxCDQP(maxiter=12, tol=1e-5, implicit_diff=False, jit='auto', unroll='auto', verbose=0)
    
    
    @jax.jit
    def allocate_PGD(u_des, T_prev, motor_omega):
        _, T_cmd = actuator_model._optimal_control_allocation(u_des, mixer_matrix, T_prev, motor_omega)
        return T_cmd
    
    @jax.jit
    def allocate_CDQP(u_des, T_prev):
        A  = Q_u @ mixer_matrix
        H  = A.T @ A + REG * jnp.eye(4, dtype=mixer_matrix.dtype)
        f  = -(A.T @ (Q_u @ u_des) + REG * T_prev)
        T0 = jnp.clip(T_prev, actuator_model.T_min, actuator_model.T_max)
        sol = box_qp.run(init_params=T0, params_obj=(H, f),
                        params_ineq=(actuator_model.T_min, actuator_model.T_max))
        return jnp.clip(sol.params, actuator_model.T_min, actuator_model.T_max)
    
    # SOLUTION CONVERGENCE BENCHMARK
    test_cases = [
        ("A: no sat",
        jnp.array([MASS*G, 0.0, 0.0, 0.0]),        False),
        ("B: aggressive roll 2 motors sat)",
        jnp.array([MASS*G, 1.0, 0.0, 0.0]),         True),
        ("C: high thrust ",
        jnp.array([4.0*T_MAX*1.3, 0.0, 0.0, 0.0]),  True),
        ("D: high neg pitch",
        jnp.array([4.0*T_MIN*1.3, 0.0, 0.0, 0.0]),  True),
        ("E: combined",
        jnp.array([MASS*G*0.5, 2.0, 2.0, 0.3]),     True),
    ]
    
    print("-" * 50)
    print("[ACTUATOR_MODEL] Convergence benchmark")
    print("-" * 50)
    for label, u_des, _ in test_cases:
        T_max = jnp.ones(4)*T_MAX
        T_min = jnp.ones(4)*T_MIN
        T_naive = jnp.linalg.inv(mixer_matrix)@u_des
        T_pgd = allocate_PGD(u_des, T_prev, W_HOVER)
        T_box = allocate_CDQP(u_des, T_prev)

        n_sat_pgd = int(jnp.sum((T_pgd > T_MAX-0.01) | (T_pgd < T_MIN+0.01)))
        n_sat_box = int(jnp.sum((T_box > T_MAX-0.01) | (T_box < T_MIN+0.01)))

        u_pgd = mixer_matrix @ T_pgd
        u_box = mixer_matrix @ T_box

        print(f"\nTEST {label}")
        print(f"u_des = {jnp.round(u_des, 3)}")
        print(f"T_naive = {jnp.round(T_naive, 4)}")
        print(f"T_max = {jnp.round(T_max, 4)}")
        print(f"T_max = {jnp.round(T_min, 4)}")
        print(f"T_pgd = {jnp.round(T_pgd, 4)}  ({n_sat_pgd} motor(s) at limit)")
        print(f"T_boxcdqp = {jnp.round(T_box, 4)}  ({n_sat_box} motor(s) at limit)")
        print(f"u_pgd = {jnp.round(u_pgd, 4)}")
        print(f"u_box = {jnp.round(u_box, 4)}")
        print(f"du = {jnp.round(u_box-u_pgd, 4)}")


    print("\n" + "-" * 50)
    print("[ACTUATOR_MODEL] Batched speed benchmark")
    print("-" * 50)
    
    B           = 2048
    u_batch     = jnp.tile(jnp.array([MASS*G*1.5, 2.0, 2.0, 0.3]), (B, 1))
    T_batch     = jnp.tile(T_prev,       (B, 1))
    omega_batch = jnp.tile(W_HOVER,  (B, 1))

    pgd_vmap = jax.jit(jax.vmap(allocate_PGD,      in_axes=(0, 0, 0)))
    box_vmap = jax.jit(jax.vmap(allocate_CDQP,  in_axes=(0, 0)))

    pgd_vmap(u_batch, T_batch, omega_batch).block_until_ready()
    box_vmap(u_batch, T_batch).block_until_ready()

    N_REPS = 1000
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        pgd_vmap(u_batch, T_batch, omega_batch).block_until_ready()
    pgd_ms = (time.perf_counter() - t0) / N_REPS * 1000

    t0 = time.perf_counter()
    for _ in range(N_REPS):
        box_vmap(u_batch, T_batch).block_until_ready()
    box_ms = (time.perf_counter() - t0) / N_REPS * 1000

    speedup = box_ms / pgd_ms
    print(f"PGD      (12 iters)  — {B} envs:  {pgd_ms:.3f} ms")
    print(f"BoxCDQP  (12 iters)  — {B} envs:  {box_ms:.3f} ms")
    print(f"Speedup: {speedup:.2f}")
