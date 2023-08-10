import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP

# Custom imports:
import optimization_utilities
import controller_utilities
import geometry_utilities
from base_controller import BaseController

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

        # Needed for Simulator Initialization:
        self.task_subspace = jnp.array(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
            ],
            dtype=dtype,
        ).T
        self.num_scales_task = self.task_subspace.shape[1]
        self.postural_subspace = jnp.ones(
            (self.plant.num_velocities(), 1), dtype=dtype,
        )
        self.num_scales_posture = self.postural_subspace.shape[1]
        self.limits_q = (jnp.asarray(self.plant_limits.q.lower), jnp.asarray(self.plant_limits.q.upper))
        self.limits_v = (jnp.asarray(self.plant_limits.v.lower), jnp.asarray(self.plant_limits.v.upper))
        self.limits_acceleration = (jnp.asarray(self.plant_limits.vd.lower), jnp.asarray(self.plant_limits.vd.upper))
        self.limits_control = (jnp.asarray(self.plant_limits.u.lower), jnp.asarray(self.plant_limits.u.upper))
        self.q_scale = 20.0
        self.v_scale = 10.0

    def preprocess(self):
        # Isolate Functions:
        self.constraint_function = lambda x, y, z, i, j, k, l, m, n: optimization_utilities.calculate_constraints(
            x, y, z, i, j, k, l, m, n,
            self.limits_q,
            self.limits_v,
            self.limits_acceleration,
            self.limits_control,
            self.task_subspace,
            self.postural_subspace,
            self.acceleration_bounds_dt,
            self.q_scale,
            self.v_scale,
        )

        self.objective_function = lambda x, y: optimization_utilities.calculate_objective(
            x, y,
            self.posture_weight,
            self.num_scales_task,
            self.num_scales_posture,
        )

    # Keep name the same
    def calc_control(
        self,
        t,
        pose_actual,
        pose_desired,
        q0,
    ):
        # Positions and Velocities:
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        # Calculate Dynamics:
        mass_matrix, coriolis_terms, tau_g = controller_utilities.calculate_dynamics(
            self.plant, self.context,
        )
        mass_matrix_inverse = np.linalg.inv(mass_matrix)
        bias_term = coriolis_terms - tau_g

        # Unpack:
        X, V, task_jacobian, task_jacobian_dv = pose_actual
        packed_output = controller_utilities.reproject_mass(
            mass_matrix_inverse, task_jacobian,
        )
        mass_task_projection, mass_task_projection_inverse, \
            task_jacobian, task_jacobian_bar, task_nullspace_transpose = packed_output
        task_nullspace = task_nullspace_transpose.T

        # Compute spatial feedback:
        kp_t, kd_t = self.gains.task
        X_des, V_des, A_des = pose_desired
        V_des = V_des.get_coeffs()
        A_des = A_des.get_coeffs()
        e_t = geometry_utilities.se3_vector_minus(X, X_des)
        ed_t = V - V_des
        desired_task_acceleration = A_des - kp_t * e_t - kd_t * ed_t

        # Compute posture feedback:
        kp_p, kd_p = self.gains.posture
        e_p = q - q0
        e_p_dir = controller_utilities.vec_dot_norm(e_p, task_nullspace @ e_p)
        e_p *= e_p_dir
        ed_p = v
        desired_postural_acceleration = -kp_p * e_p - kd_p * ed_p

        task_projection = task_jacobian.T @ mass_task_projection
        postural_projection = task_nullspace_transpose @ mass_matrix

        desired_task_scales = np.ones(self.num_scales_task)
        desired_postural_scales = np.ones(self.num_scales_posture)

        # Compute Constraints:
        qp_constraints, control_constraints = self.constraint_function(
            q,
            v,
            task_projection,
            postural_projection,
            desired_task_acceleration,
            desired_postural_acceleration,
            mass_matrix_inverse,
            bias_term,
            task_jacobian_dv,
        )

        Q, c = self.objective_function(
            desired_task_scales,
            desired_postural_scales,
        )

        # BUILD PROGRAM:
        A, lb, ub = qp_constraints
        A_control, b_control = control_constraints

        qp = BoxOSQP(
            momentum=1.6,
            rho_start=1e-1,
            primal_infeasible_tol=1e-8,
            dual_infeasible_tol=1e-8,
            maxiter=5000,
            tol=1e-2,
            termination_check_frequency=25,
            verbose=0,
            jit=False,
        )

        sol, state = qp.run(
            params_obj=(Q, c),
            params_eq=A,
            params_ineq=(lb, ub),
        )

        print(f"Optimization Status: {state.status}")

        task_scales = sol.primal[:self.num_scales_task]
        postural_scales = sol.primal[self.num_scales_task:]

        self.prev_sol = sol.primal

        control_multiplier = np.concatenate([task_scales, postural_scales])

        tau = A_control @ control_multiplier + b_control
        tau = self.plant_limits.u.saturate(tau)

        return tau
