from typing import Union

import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP
import matplotlib.pyplot as plt

# Custom imports:
import optimization_utilities
import controller_utilities
import geometry_utilities
from base_controller import BaseController
import plot_utilities

import time

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


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

        self.should_save = True
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
        self.limits_q = (
            jnp.asarray(self.plant_limits.q.lower), jnp.asarray(self.plant_limits.q.upper)
        )
        self.limits_v = (
            jnp.asarray(self.plant_limits.v.lower), jnp.asarray(self.plant_limits.v.upper)
        )
        self.limits_acceleration = (
            jnp.asarray(self.plant_limits.vd.lower), jnp.asarray(self.plant_limits.vd.upper)
        )
        self.limits_control = (
            jnp.asarray(self.plant_limits.u.lower), jnp.asarray(self.plant_limits.u.upper)
        )
        self.q_scale = 20.0
        self.v_scale = 10.0
        self.desired_task_scales = jnp.ones(self.num_scales_task)
        self.desired_postural_scales = jnp.ones(self.num_scales_posture)
        self.qp = BoxOSQP(
            check_primal_dual_infeasability=False,
            momentum=1.6,
            rho_start=1e-6,
            primal_infeasible_tol=1e-8,
            dual_infeasible_tol=1e-8,
            maxiter=200,
            tol=1e-5,
            termination_check_frequency=5,
            verbose=0,
            implicit_diff=False,
            jit=True,
            unroll=True,
        )

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
        # Start Timer:
        start_time = time.time()

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
        pose, velocity, task_jacobian, task_jacobian_dv = pose_actual
        packed_output = controller_utilities.reproject_mass(
            mass_matrix_inverse, task_jacobian,
        )
        mass_task_projection, mass_task_projection_inverse, \
            task_jacobian, task_jacobian_bar, task_nullspace_transpose = packed_output
        task_nullspace = task_nullspace_transpose.T

        # Compute spatial feedback:
        kp_t, kd_t = self.gains.task
        desired_pose, desired_velocity, desired_acceleration = pose_desired
        desired_velocity = desired_velocity.get_coeffs()
        desired_acceleration = desired_acceleration.get_coeffs()
        task_pose_error = geometry_utilities.se3_vector_minus(
            pose, desired_pose,
        )
        task_velocity_error = velocity - desired_velocity
        desired_task_acceleration = (
            desired_acceleration - kp_t * task_pose_error - kd_t * task_velocity_error
        )

        # Compute posture feedback:
        kp_p, kd_p = self.gains.posture
        postural_pose_error = q - q0
        postural_pose_scale = controller_utilities.vec_dot_norm(
            postural_pose_error, task_nullspace @ postural_pose_error,
        )
        postural_pose_error *= postural_pose_scale
        postural_velocity_error = v
        desired_postural_acceleration = (
            -kp_p * postural_pose_error - kd_p * postural_velocity_error
        )

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

        sol, state = self.qp.run(
            init_params=self.prev_sol,
            params_obj=(Q, c),
            params_eq=A,
            params_ineq=(lb, ub),
        )

        task_scales = sol.primal[0][:self.num_scales_task]
        postural_scales = sol.primal[0][self.num_scales_task:]

        self.prev_sol = sol
        self.prev_status = state.status

        control_multiplier = np.concatenate([task_scales, postural_scales])

        tau = A_control @ control_multiplier + b_control
        tau = self.plant_limits.u.saturate(tau)

        elapsed_time = time.time() - start_time
        print(f"Optimization Status: {state.status} \t Control: {tau} \t Time: {elapsed_time}")

        def save_data():
            edd_c_p_null = mass_matrix_inverse @ task_nullspace_transpose @ mass_matrix @ desired_postural_acceleration
            _, sigmas, _ = np.linalg.svd(task_jacobian)
            self.ts.append(t)
            self.qs.append(q)
            self.vs.append(v)
            self.us.append(tau)
            self.edd_ts.append(desired_task_acceleration)
            self.s_ts.append(task_scales)
            self.r_ts.append(np.zeros(6))
            self.edd_ps.append(desired_postural_acceleration)
            self.edd_ps_null.append(edd_c_p_null)
            self.s_ps.append(postural_scales)
            self.limits_infos.append(0)
            self.sigmas.append(sigmas)

        if self.should_save:
            save_data()

        return tau

    # # Keep name the same
    # def calc_control(
    #     self,
    #     t,
    #     pose_actual,
    #     pose_desired,
    #     q0,
    # ):
    #     def func(
    #         q: jax.Array,
    #         v: jax.Array,
    #         mass_matrix: jax.Array,
    #         coriolis_terms: jax.Array,
    #         tau_g: jax.Array,
    #         task_pose_error: jax.Array,
    #         velocity: jax.Array,
    #         desired_velocity: jax.Array,
    #         desired_acceleration: jax.Array,
    #         task_jacobian: jax.Array,
    #         task_jacobian_dv: jax.Array,
    #         desired_task_scales: jax.Array,
    #         desired_postural_scales: jax.Array,
    #         previous_solution: Union[None, jax.Array],
    #         kp_task: float,
    #         kd_task: float,
    #         kp_posture: float,
    #         kd_posture: float,
    #         identity: jax.Array,
    #     ):
    #         # Calculate Dynamics:
    #         mass_matrix_inverse = jnp.linalg.inv(mass_matrix)
    #         bias_term = coriolis_terms - tau_g

    #         # Reproject Mass:
    #         packed_output = controller_utilities.jax_reproject_mass(
    #             mass_matrix_inverse, task_jacobian, identity,
    #         )
    #         mass_task_projection, task_jacobian, task_nullspace_transpose = packed_output
    #         task_nullspace = task_nullspace_transpose.T

    #         # Task Feedback:
    #         task_velocity_error = velocity - desired_velocity
    #         desired_task_acceleration = (
    #             desired_acceleration - kp_task * task_pose_error - kd_task * task_velocity_error
    #         )

    #         # Posture Feedback:
    #         postural_pose_error = q - q0
    #         postural_pose_error_scale = controller_utilities.jax_vec_dot_norm(
    #             postural_pose_error, task_nullspace @ postural_pose_error,
    #         )
    #         postural_pose_error *= postural_pose_error_scale
    #         postural_velocity_error = v
    #         desired_postural_acceleration = (
    #             -kp_posture * postural_pose_error - kd_posture * postural_velocity_error
    #         )

    #         #
    #         task_projection = task_jacobian.T @ mass_task_projection
    #         postural_projection = task_nullspace_transpose @ mass_matrix

    #         # Compute Constraints:
    #         qp_constraints, control_constraints = self.constraint_function(
    #             q,
    #             v,
    #             task_projection,
    #             postural_projection,
    #             desired_task_acceleration,
    #             desired_postural_acceleration,
    #             mass_matrix_inverse,
    #             bias_term,
    #             task_jacobian_dv,
    #         )

    #         Q, c = self.objective_function(
    #             desired_task_scales,
    #             desired_postural_scales,
    #         )

    #         # BUILD PROGRAM:
    #         A, lb, ub = qp_constraints
    #         A_control, b_control = control_constraints

    #         sol, state = self.qp.run(
    #             init_params=previous_solution,
    #             params_obj=(Q, c),
    #             params_eq=A,
    #             params_ineq=(lb, ub),
    #         )

    #         task_scales = sol.primal[0][:self.num_scales_task]
    #         postural_scales = sol.primal[0][self.num_scales_task:]

    #         control_multiplier = jnp.concatenate([task_scales, postural_scales])

    #         tau = A_control @ control_multiplier + b_control

    #         return tau, sol, state

    #     # Start Timer:
    #     start_time = time.time()

    #     # Positions and Velocities:
    #     q = self.plant.GetPositions(self.context)
    #     v = self.plant.GetVelocities(self.context)

    #     # Calculate Dynamics:
    #     mass_matrix, coriolis_terms, tau_g = controller_utilities.calculate_dynamics(
    #         self.plant, self.context,
    #     )

    #     # Unpack inputs and gains:
    #     kp_task, kd_task = self.gains.task
    #     kp_posture, kd_posture = self.gains.posture
    #     pose, velocity, task_jacobian, task_jacobian_dv = pose_actual
    #     desired_pose, desired_velocity, desired_acceleration = pose_desired
    #     desired_velocity = desired_velocity.get_coeffs()
    #     desired_acceleration = desired_acceleration.get_coeffs()
    #     task_pose_error = geometry_utilities.se3_vector_minus(
    #         pose, desired_pose,
    #     )
    #     identity = jnp.eye(task_jacobian.shape[-1])

    #     tau, solution, state = jax.jit(func)(
    #         q,
    #         v,
    #         mass_matrix,
    #         coriolis_terms,
    #         tau_g,
    #         task_pose_error,
    #         velocity,
    #         desired_velocity,
    #         desired_acceleration,
    #         task_jacobian,
    #         task_jacobian_dv,
    #         self.desired_task_scales,
    #         self.desired_postural_scales,
    #         self.prev_sol,
    #         kp_task,
    #         kd_task,
    #         kp_posture,
    #         kd_posture,
    #         identity,
    #     )

    #     # Warm Start:
    #     if state.status == 1:
    #         self.prev_sol = solution
    #     else:
    #         self.prev_sol = None

    #     tau = self.plant_limits.u.saturate(tau)

    #     elapsed_time = time.time() - start_time
    #     print(f"Optimization Status: {state.status} \t Control: {tau} \t Time: {elapsed_time}")

    #     return tau

    def show_plots(self):
        ts = np.array(self.ts)
        qs = np.array(self.qs)
        vs = np.array(self.vs)
        us = np.array(self.us)
        edd_ts = np.array(self.edd_ts)
        s_ts = np.array(self.s_ts)
        r_ts = np.array(self.r_ts)
        edd_ps = np.array(self.edd_ps)
        edd_ps_null = np.array(self.edd_ps_null)
        s_ps = np.array(self.s_ps)
        sigmas = np.array(self.sigmas)

        sub = slice(None, None)
        sel = (slice(None, None), sub)

        def plot_lim(limits):
            lower, upper = limits
            lower = lower[sub]
            upper = upper[sub]
            ts_lim = ts[[0, -1]]
            plot_utilities.reset_color_cycle()
            plt.plot(ts_lim, [lower, lower], ":")
            plot_utilities.reset_color_cycle()
            plt.plot(ts_lim, [upper, upper], ":")

        _, axs = plt.subplots(num=1, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, qs[sel])
        plot_utilities.legend_for(qs[sel])
        plot_lim(self.plant_limits.q)
        plt.title("q")
        plt.sca(axs[1])
        plt.plot(ts, vs[sel])
        plot_utilities.legend_for(vs[sel])
        plot_lim(self.plant_limits.v)
        plt.title("v")
        plt.sca(axs[2])
        plt.plot(ts, us[sel])
        plot_utilities.legend_for(us[sel])
        plot_lim(self.plant_limits.u)
        plt.title("u")
        plt.tight_layout()

        _, axs = plt.subplots(num=2, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, edd_ts)
        plot_utilities.legend_for(edd_ts)
        plt.title("edd_t_c")
        plt.sca(axs[1])
        plt.plot(ts, edd_ps)
        plot_utilities.legend_for(edd_ps)
        plt.title("edd_p_c")
        plt.sca(axs[2])
        plt.plot(ts, edd_ps_null)
        plot_utilities.legend_for(edd_ps_null)
        plt.title("edd_p_c null")
        plt.tight_layout()

        _, axs = plt.subplots(num=3, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, s_ts)
        plt.ylim(-5, 5)
        plot_utilities.legend_for(s_ts)
        plt.title("s_t")
        plt.sca(axs[1])
        if len(r_ts) > 0:
            plt.plot(ts, r_ts)
            plot_utilities.legend_for(r_ts)
        plt.title("r_t")
        plt.sca(axs[2])
        if len(s_ps) > 0:
            plt.plot(ts, s_ps)
            plt.ylim(-5, 5)
            plot_utilities.legend_for(s_ps)
        plt.title("s_p")
        plt.tight_layout()

        _, axs = plt.subplots(num=4, nrows=2)
        plt.sca(axs[0])
        plt.plot(ts, sigmas)
        plot_utilities.legend_for(sigmas)
        plt.title("singular values")
        plt.sca(axs[1])
        manips = np.prod(sigmas, axis=-1)
        plt.plot(ts, manips)
        plt.title("manip index")
        plt.tight_layout()

        plt.show()
