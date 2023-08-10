import functools
from typing import Tuple, List

import numpy as np
import jax
import jax.numpy as jnp

# Custom imports:
import controller_utilities
import geometry_utilities
from base_controller import BaseController

# type aliases:
Pair = Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]
Limits = List[Pair]

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64


class ParallelController(BaseController):
    def __init__(
        self,
        plant,
        frame_W,
        frame_G,
        *,
        gains,
        plant_limits,
        acceleration_bounds_dt,
        posture_weight,
        use_torque_weights=False,
    ):
        super().__init__(plant, frame_W, frame_G)
        self.gains = gains
        self.plant_limits = plant_limits

        self.plant_ad = self.plant.ToAutoDiffXd()
        self.context_ad = self.plant_ad.CreateDefaultContext()

        self.acceleration_bounds_dt = acceleration_bounds_dt
        self.posture_weight = posture_weight
        self.use_torque_weights = use_torque_weights
        self.prev_sol = None

        self.should_save = False
        self.ts = []
        self.qs = []
        self.vs = []
        self.us = []
        self.edd_ts = []
        self.s_ts = []
        self.r_ts = []
        self.edd_ps = []
        self.edd_ps_null = []
        self.s_ps = []
        # self.r_ps = []
        self.limits_infos = []
        self.sigmas = []
        self.prev_dir = None
        self.Jmu_prev = None


    def calc_control(self, t, pose_actual, pose_desired, q0):
        # Positions and Velocities:
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        num_v = len(v)
        M, C, tau_g = controller_utilities.calculate_dynamics(self.plant, self.context)
        M_inv = np.linalg.inv(M)
        H = C - tau_g

        # Base QP formulation: (QP FORM STARTS HERE)
        eye_v = np.eye(self.num_q)
        zeros_v = np.zeros(self.num_q)
        prog = MathematicalProgram()

        X, V, J_task, J_task_dv = pose_actual
        M_task, M_task_inv, J_task, J_task_bar, N_task_transpose = controller_utilities.reproject_mass(M_inv, J_task)
        N_task = N_task_transpose.T

        # Compute spatial feedback.
        kp_task, kd_task = self.gains.task
        num_t = 6
        eye_task = np.eye(num_t)
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e_task = geometry_utilities.se3_vector_minus(X, X_des)
        de_task = V - V_des
        dde_task_c = A_des - kp_task * e_task - kd_task * de_task

        # Compute posture feedback.
        kp_posture, kd_posture = self.gains.posture
        e_posture = q - q0
        e_posture_dir = controller_utilities.vec_dot_norm(e_posture, N_task @ e_posture)
        e_posture *= e_posture_dir
        de_posture = v
        dde_posture_c = -kp_posture * e_posture - kd_posture * de_posture

        # ?
        scale_A_task = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ]).T

        num_scales_task = scale_A_task.shape[1]

        # Optimization variables:
        scale_vars_t = prog.NewContinuousVariables(num_scales_t, "scale_t")

        desired_scales_t = np.ones(num_scales_t)

        # Settings:
        # If True, will add (vd, tau) as decision variables, and impose
        # dynamics and task acceleration constraints. If False, will explicitly
        # project to torques / accelerations.
        implicit = False

        # If True, will also scale secondary objective (posture).
        scale_secondary = True

        # For implicit formulation.
        dup_eq_as_cost = False
        dup_scale = 0.1

        relax_primary = None
        relax_secondary = None

        kinematic = False

        scale_A_posture = np.ones((num_v, 1))
        num_scales_posture = scale_A_posture.shape[1]
        # Optimization variables:
        scale_vars_posture = prog.NewContinuousVariables(num_scales_posture, "scale_posture")

        assert self.use_torque_weights
        proj_task = J_task.T @ M_task
        proj_posture = N_task_transpose @ M

        # Primary, scale: (Acting on optimization variables)
        u_vars = scale_vars_t
        Au = proj_task @ np.diag(edd_t_c) @ scale_A_t
        bu = -proj_task @ Jtdot_v + H

        # Secondary, scale: (Acting on optimization variables)
        Au_p = proj_p @ np.diag(edd_p_c) @ scale_A_p
        u_vars = np.concatenate([u_vars, scale_vars_p])
        Au = np.hstack([Au, Au_p])

        # Acceleration is just affine transform.
        vd_vars = u_vars
        Avd = Minv @ Au
        bvd = Minv @ (bu - H)

        # Add limits.
        vd_limits = self.plant_limits.vd

        dt = self.acceleration_bounds_dt  # HACK
        limit_info = add_plant_limits_to_qp(
            plant_limits=self.plant_limits,
            vd_limits=vd_limits,
            dt=dt,
            q=q,
            v=v,
            prog=prog,
            vd_vars=vd_vars,
            Avd=Avd,
            bvd=bvd,
            u_vars=u_vars,
            Au=Au,
            bu=bu,
        )

        controller_utilities.add_2norm_decoupled(
            prog,
            np.ones(num_scales_t),
            desired_scales_t,
            scale_vars_t,
        )

        desired_scales_p = np.ones(num_scales_p)

        controller_utilities.add_2norm_decoupled(
            prog,
            self.posture_weight * np.ones(num_scales_p),
            self.posture_weight * desired_scales_p,
            scale_vars_p,
        )

        # Solve.
        try:
            result = solve_or_die(
                self.solver, self.solver_options, prog, x0=self.prev_sol
            )
        except RuntimeError:
            raise

        tol = 1e-8

        scale_t = result.GetSolution(scale_vars_t)
        relax_t = np.zeros(num_t)
        scale_p = result.GetSolution(scale_vars_p)

        infeas = result.GetInfeasibleConstraintNames(prog, tol=tol)
        infeas_text = "\n" + indent("\n".join(infeas), "  ")
        assert len(infeas) == 0, infeas_text
        self.prev_sol = result.get_x_val()

        u_mul = scale_t
        if scale_secondary:
            u_mul = np.concatenate([u_mul, scale_p])
        if relax_primary is not None:
            u_mul = np.concatenate([u_mul, relax_t])
        tau = Au @ u_mul + bu

        tau = self.plant_limits.u.saturate(tau)

        edd_c_p_null = Minv @ Nt_T @ M @ edd_p_c
        _, sigmas, _ = np.linalg.svd(Jt)

        if self.should_save:
            self.ts.append(t)
            self.qs.append(q)
            self.vs.append(v)
            self.us.append(tau)
            self.edd_ts.append(edd_t_c)
            self.s_ts.append(scale_t)
            self.r_ts.append(relax_t)
            self.edd_ps.append(edd_p_c)
            self.edd_ps_null.append(edd_c_p_null)
            if scale_secondary:
                self.s_ps.append(scale_p)
            self.limits_infos.append(limit_info)
            self.sigmas.append(sigmas)

        return tau

    def qp_preprocess(self):
        # Static Arguments:
        v = self.plant.GetVelocities(self.context)
        self.num_v = len(v)
        self.limits = {
                "q": self.plant_limits.q,
                "v": self.plant_limits.v,
                "acceleration": self.plant_limits.vd,
                "control": self.plant_limits.u,
        }
        self.task_subspace = jnp.array(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
            ],
            dtype=dtype,
        ).T
        self.num_scales_task = self.task_subspace.shape[1]
        self.postural_subspace = jnp.ones((self.num_v, 1), dtype=dtype)
        self.num_scales_posture = self.postural_subspace.shape[1]
        self.q_scale = 20.0
        self.v_scale = 10.0

        # Isolate Functions:
        self.constraint_function = lambda x, y, z, i, j, k, l, m, n: calculate_constraints(
            x, y, z, i, j, k, l, m, n,
            self.limits,
            self.task_subspace,
            self.postural_subspace,
            self.acceleration_bounds_dt,
            self.q_scale,
            self.v_scale,
        )

        self.objective_function = lambda x, y: calculate_objective(
            x, y,
            self.posture_weight,
            self.num_scales_task,
            self.num_scales_posture,
        )

    def calculate_control(
        self,
        pose_actual, 
        pose_desired,
        q0,
    ):
        # Positions and Velocities:
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        # Calculate Dynamics:
        M, C, tau_g = controller_utilities.calculate_dynamics(self.plant, self.context)
        M_inv = np.linalg.inv(M)
        H = C - tau_g

        # Unpack:
        X, V, J_task, J_task_dv = pose_actual
        M_task, M_task_inv, J_task, J_task_bar, N_task_transpose = controller_utilities.reproject_mass(M_inv, J_task)
        N_task = N_task_transpose.T

        # Compute spatial feedback.
        kp_t, kd_t = self.gains.task
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e_t = se3_vector_minus(X, X_des)
        ed_t = V - V_des
        edd_t_c = A_des - kp_t * e_t - kd_t * ed_t

        # Compute posture feedback.
        kp_p, kd_p = self.gains.posture
        e_p = q - q0
        e_p_dir = vec_dot_norm(e_p, Nt @ e_p)
        e_p *= e_p_dir  # From OSC hacking.
        ed_p = v
        edd_p_c = -kp_p * e_p - kd_p * ed_p


    # Push this to a utilities file.
    @functools.partial(
        jax.jit,
        static_argnames=[
            "task_subspace", "postural_subspace", "dt", "q_scale", "v_scale"
        ],
    )
    def calculate_constraints(
        q: jax.Array,
        v: jax.Array,
        task_projection: jax.Array,
        postural_projection: jax.Array,
        desired_task_acceleration: jax.Array,
        desired_postural_acceleration: jax.Array,
        mass_matrix_inverse: jax.Array,
        bias_term: jax.Array,
        J_task_dv: jax.Array,
        limits: Limits,
        task_subspace: jax.Array,
        postural_subspace: jax.Array,
        dt: float,
        q_scale: float = 20.0,
        v_scale: float = 10.0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Unpack Limits:
        q_min, q_max = limits['q']  # type: ignore
        v_min, v_max = limits['v']  # type: ignore
        vd_min, vd_max = limits['acceleration']  # type: ignore
        u_min, u_max = limits['control']  # type: ignore

        # Task Constraint Matrix:
        A_task = task_projection @ jnp.diag(desired_task_acceleration) @ task_subspace
        b = -task_projection @ J_task_dv + bias_term

        # Posture Constraint Matrix:
        A_posture = postural_projection @ jnp.diag(desired_postural_acceleration) @ postural_subspace

        # Combine:
        A_combined = jnp.hstack([A_task, A_posture])

        # Acceleration Constraints:
        A_acceleration = mass_matrix_inverse @ A_combined
        b_acceleration = mass_matrix_inverse @ (b - bias_term)

        # CBF Constraints:
        kq_1 = 2 / (dt ** 2)
        kq_2 = 2 / dt
        kv_1 = 1 / dt
        kq_1 /= (q_scale ** 2)
        kq_2 /= v_scale
        kv_1 /= v_scale
        calculate_b_q = lambda x, y: -kq_1 * x - kq_2 * y
        calculate_b_v = lambda x: -kv_1 * x

        # q_min
        h_q_min = q - q_min
        hd_q_min = v
        b_q_min = calculate_b_q(h_q_min, hd_q_min)

        # q_max
        h_q_max = q_max - q
        hd_q_max = -v
        b_q_max = calculate_b_q(h_q_max, hd_q_max)

        # v_min
        h_v_min = v - v_min
        b_v_min = calculate_b_v(h_v_min)

        # v_max
        h_v_max = v_max - v
        b_v_max = calculate_b_v(h_v_max)

        A = jnp.vstack(
            [A_acceleration, A_acceleration, A_acceleration, A_combined],
        )
        l = jnp.concatenate([
            b_q_min - b_acceleration,
            b_v_min - b_acceleration,
            vd_min - b_acceleration,
            u_min - b,
        ])
        u = jnp.concatenate([
            -b_q_max - b_acceleration,
            -b_v_max - b_acceleration,
            vd_max - b_acceleration,
            u_max - b,
        ])
        return A, l, u
    
    #TODO: All arguments are static?
    @functools.partial(
        jax.jit,
        static_argnames=[
            "posture_weight", "num_scales_task", "num_scales_posture"
        ],
    )
    def calculate_objective(
        desired_scales_task: jax.Array,
        desired_scales_posture: jax.Array,
        posture_weight: float,
        num_scales_task: int,
        num_scales_posture: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Calculate Objective Function:
        Q_task, c_task = controller_utilities.calculate_quadratic_cost(
            jnp.ones(num_scales_task),
            desired_scales_task,
        )
        Q_posture, c_posture = controller_utilities.calculate_quadratic_cost(
            posture_weight * jnp.ones(num_scales_posture),
            posture_weight * desired_scales_posture,
        )
        # Combine:
        Q = jnp.block([
            [Q_task, jnp.zeros((num_scales_task, num_scales_posture))],
            [jnp.zeros((num_scales_posture, num_scales_task)), Q_posture],
        ])
        c = jnp.concatenate([c_task, c_posture])
        return Q, c


def test_constraints():
    # Dummy Inputs:
    dummy_q = np.zeros(7)
    dummy_v = np.zeros(7)

    dummy_task_projection = np.zeros((7, 6))
    dummy_postural_projection = np.zeros((7, 7))

    desired_task_acceleration = np.zeros((6,))
    desired_postural_acceleration = np.zeros((7,))

    task_subsapce = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ]).T
    postural_subspace = np.ones((7, 1))

    mass_matrix_inverse = np.eye(7)
    bias_term = np.zeros((7,))

    J_task_dv = np.zeros((6,))

    dt = 0.02
    limits = {
        "q": (jnp.zeros((7,)), jnp.zeros((7,))),
        "v": (jnp.zeros((7,)), jnp.zeros((7,))),
        "acceleration": (jnp.zeros((7,)), jnp.zeros((7,))),
        "control": (jnp.zeros((7,)), jnp.zeros((7,))),
    }

    # Run:
    A, l, u = calculate_constraints(
        dummy_q,
        dummy_v,
        dummy_task_projection,
        dummy_postural_projection,
        desired_task_acceleration,
        desired_postural_acceleration,
        task_subsapce,
        postural_subspace,
        mass_matrix_inverse,
        bias_term,
        J_task_dv,
        limits,
        dt,
    )


def test_objective():
    # Dummy Inputs:
    desired_scales_task = np.ones((2,))
    desired_scales_posture = np.ones((1,))
    posture_weight = 1.0
    num_scales_task = 2
    num_scales_posture = 1

    # Run:
    Q, c = calculate_objective(
        desired_scales_task,
        desired_scales_posture,
        posture_weight,
        num_scales_task,
        num_scales_posture,
    )


def main():
    test_constraints()
    test_objective()


if __name__ == "__main__":
    main()
