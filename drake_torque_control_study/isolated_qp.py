from typing import Union, Callable
import functools

import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP

# Custom imports:
import controller_utilities
import geometry_utilities
import optimization_utilities

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


def get_drake_values(
    plant,
    context,
    pose_actual,
    pose_desired,
    gains,
):
    # Positions and Velocities:
    q = plant.GetPositions(context)
    v = plant.GetVelocities(context)

    # Calculate Dynamics:
    mass_matrix, coriolis_terms, tau_g = controller_utilities.calculate_dynamics(
        plant, context,
    )

    # Unpack inputs and gains:
    kp_task, kd_task = gains.task
    kp_posture, kd_posture = gains.posture
    pose, velocity, task_jacobian, task_jacobian_dv = pose_actual
    desired_pose, desired_velocity, desired_acceleration = pose_desired
    desired_velocity = desired_velocity.get_coeffs()
    desired_acceleration = desired_acceleration.get_coeffs()
    task_pose_error = geometry_utilities.se3_vector_minus(
        pose, desired_pose,
    )
    identity = jnp.eye(task_jacobian.shape[-1])
    return (q, v, mass_matrix, coriolis_terms, tau_g, task_pose_error,
            velocity, desired_velocity, desired_acceleration, task_jacobian,
            task_jacobian_dv, kp_task, kd_task, kp_posture, kd_posture, identity)


@functools.partial(jax.jit, static_argnames=["num_scales_task"])
def calculate_control(
    q: jax.Array,
    v: jax.Array,
    initial_q: jax.Array,
    mass_matrix: jax.Array,
    coriolis_terms: jax.Array,
    tau_g: jax.Array,
    task_pose_error: jax.Array,
    velocity: jax.Array,
    desired_velocity: jax.Array,
    desired_acceleration: jax.Array,
    task_jacobian: jax.Array,
    task_jacobian_dv: jax.Array,
    desired_task_scales: jax.Array,
    desired_postural_scales: jax.Array,
    kp_task: float,
    kd_task: float,
    kp_posture: float,
    kd_posture: float,
    identity: jax.Array,
    num_scales_task: int,
    limits_q,
    limits_v,
    limits_acceleration,
    limits_control,
    task_subspace,
    postural_subspace,
    acceleration_bounds_dt,
    q_scale,
    v_scale,
):
    # Calculate Dynamics:
    mass_matrix_inverse = jnp.linalg.inv(mass_matrix)
    bias_term = coriolis_terms - tau_g

    # Reproject Mass:
    packed_output = controller_utilities.jax_reproject_mass(
        mass_matrix_inverse, task_jacobian, identity,
    )
    mass_task_projection, task_jacobian, task_nullspace_transpose = packed_output
    task_nullspace = task_nullspace_transpose.T

    # Task Feedback:
    task_velocity_error = velocity - desired_velocity
    desired_task_acceleration = (
        desired_acceleration - kp_task * task_pose_error - kd_task * task_velocity_error
    )

    # Posture Feedback:
    postural_pose_error = q - initial_q
    postural_pose_error_scale = controller_utilities.jax_vec_dot_norm(
        postural_pose_error, task_nullspace @ postural_pose_error,
    )
    postural_pose_error *= postural_pose_error_scale
    postural_velocity_error = v
    desired_postural_acceleration = (
        -kp_posture * postural_pose_error - kd_posture * postural_velocity_error
    )

    #
    task_projection = task_jacobian.T @ mass_task_projection
    postural_projection = task_nullspace_transpose @ mass_matrix

    # Compute Constraints:
    qp_constraints, control_constraints = optimization_utilities.calculate_constraints(
        q,
        v,
        task_projection,
        postural_projection,
        desired_task_acceleration,
        desired_postural_acceleration,
        mass_matrix_inverse,
        bias_term,
        task_jacobian_dv,
        limits_q,
        limits_v,
        limits_acceleration,
        limits_control,
        task_subspace,
        postural_subspace,
        acceleration_bounds_dt,
        q_scale,
        v_scale,
    )

    Q, c = optimization_utilities.calculate_objective(
        desired_task_scales,
        desired_postural_scales,
        1.0,
        2,
        1,
    )

    # BUILD PROGRAM:
    A, lb, ub = qp_constraints
    A_control, b_control = control_constraints

    qp = BoxOSQP(
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

    sol, state = qp.run(
        None,
        (Q, c),
        A,
        (lb, ub),
    )

    task_scales = sol.primal[0][:num_scales_task]
    postural_scales = sol.primal[0][num_scales_task:]

    control_multiplier = jnp.concatenate([task_scales, postural_scales])

    tau = A_control @ control_multiplier + b_control

    return tau, sol, state


@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0), out_axes=(0, 0))
def vmapped_run(self, Q, c, A, lb, ub):
    sol, state = self.qp.run(
        init_params=self.prev_sol,
        params_obj=(Q, c),
        params_eq=A,
        params_ineq=(lb, ub),
    )
    return sol, state