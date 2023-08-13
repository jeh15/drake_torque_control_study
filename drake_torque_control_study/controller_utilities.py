from typing import Tuple
import functools

import numpy as np
import jax
import jax.numpy as jnp

from drake_torque_control_study.multibody_extras import calc_velocity_jacobian


def calc_spatial_values(plant, context, frame_W, frame_G):
    X = plant.CalcRelativeTransform(context, frame_W, frame_G)
    J, Jdot_v = calc_velocity_jacobian(
        plant, context, frame_W, frame_G, include_bias=True
    )
    v = plant.GetVelocities(context)
    V = J @ v
    return X, V, J, Jdot_v


def calculate_dynamics(
        plant,
        context,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = plant.CalcMassMatrix(context)
    C = plant.CalcBiasTerm(context)
    tau_g = plant.CalcGravityGeneralizedForces(context)
    return M, C, tau_g


def reproject_mass(M_inv, J_task):
    _, num_dof = J_task.shape
    I_dyn = np.eye(num_dof)

    # Maps from task forces to task accelerations.
    M_task_inv = J_task @ M_inv @ J_task.T

    # Maps from task accelerations to task forces.
    M_task = np.linalg.inv(M_task_inv)

    # Maps from task accelerations to generalized accelerations.
    # Transpose maps from generalized forces to task forces.
    J_taskbar = M_inv @ J_task.T @ M_task

    # Generalized force nullspace.
    N_task_transpose = I_dyn - J_task.T @ J_taskbar.T
    return (M_task, M_task_inv, J_task, J_taskbar, N_task_transpose)


def jax_reproject_mass(
    mass_matrix_inverse: jax.Array,
    task_jacobian: jax.Array,
    identity: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Maps from task forces to task accelerations.
    mass_task_projection_inverse = task_jacobian @ mass_matrix_inverse @ task_jacobian.T

    # Maps from task accelerations to task forces.
    mass_task_projection = jnp.linalg.inv(mass_task_projection_inverse)

    # Maps from task accelerations to generalized accelerations.
    # Transpose maps from generalized forces to task forces.
    task_jacobian_bar = mass_matrix_inverse @ task_jacobian.T @ mass_task_projection

    # Generalized force nullspace.
    task_nullspace_transpose = identity - task_jacobian.T @ task_jacobian_bar.T
    return (mass_task_projection, task_jacobian, task_nullspace_transpose)


def vec_dot_norm(a, b, *, tol=1e-8):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n <= tol:
        return 0.0
    else:
        # arcos of this value gives angle.
        return a.dot(b) / n


def jax_vec_dot_norm(a: jax.Array, b: jax.Array, tol: float = 1e-8):
    n = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    return jnp.where(n <= tol, 0.0, a.dot(b) / n)


@jax.jit
def calculate_quadratic_cost(a: jax.Array, b: jax.Array):
    Q = 2 * jnp.diag(a * a)
    c = -2 * a * b
    return Q, c
