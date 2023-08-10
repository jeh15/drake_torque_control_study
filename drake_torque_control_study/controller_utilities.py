from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp

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
    return M_task, M_task_inv, J_task, J_taskbar, N_task_transpose


def vec_dot_norm(a, b, *, tol=1e-8):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n <= tol:
        return 0.0
    else:
        # arcos of this value gives angle.
        return a.dot(b) / n


def add_plant_limits_to_qp(
    *,
    plant_limits,
    vd_limits,
    dt,
    q,
    v,
    prog,
    vd_vars,
    Avd,
    bvd,
    u_vars,
    Au,
    bu,
):
    q_min, q_max = plant_limits.q
    v_min, v_max = plant_limits.v

    num_v = len(v)
    Iv = np.eye(num_v)

    # CBF formulation.
    # Goal: h >= 0 for all admissible states
    # hdd = c*vd >= -k_1*h -k_2*hd = b

    # Gains corresponding to naive formulation.
    aq_1 = lambda x: x
    aq_2 = aq_1
    av_1 = lambda x: x
    kq_1 = 2 / (dt * dt)
    kq_2 = 2 / dt
    kv_1 = 1 / dt

    q_dt_scale = 20
    v_dt_scale = 10
    kq_1 /= q_dt_scale**2
    kq_2 /= v_dt_scale
    kv_1 /= v_dt_scale

    # q_min
    h_q_min = q - q_min
    hd_q_min = v
    c_q_min = 1
    b_q_min = -kq_1 * aq_1(h_q_min) - kq_2 * aq_2(hd_q_min)
    # q_max
    h_q_max = q_max - q
    hd_q_max = -v
    c_q_max = -1
    b_q_max = -kq_1 * aq_1(h_q_max) - kq_2 * aq_2(hd_q_max)
    # v_min
    h_v_min = v - v_min
    c_v_min = 1
    b_v_min = -kv_1 * av_1(h_v_min)
    # v_max
    h_v_max = v_max - v
    c_v_max = -1
    b_v_max = -kv_1 * av_1(h_v_max)

    # Add constraints.
    # N.B. Nominal CBFs (c*vd >= b) are lower bounds. For CBFs where c=-1,
    # we can pose those as upper bounds (vd <= -b).
    prog.AddLinearConstraint(
        Avd,
        b_q_min - bvd,
        -b_q_max - bvd,
        vd_vars,
    ).evaluator().set_description("pos cbf")
    prog.AddLinearConstraint(
        Avd,
        b_v_min - bvd,
        -b_v_max - bvd,
        vd_vars,
    ).evaluator().set_description("vel cbf")

    if vd_limits.any_finite():
        vd_min, vd_max = vd_limits
        prog.AddLinearConstraint(
            Avd,
            vd_min - bvd,
            vd_max - bvd,
            vd_vars,
        ).evaluator().set_description("accel")

    # - Torque.
    if u_vars is not None and plant_limits.u.any_finite():
        u_min, u_max = plant_limits.u
        prog.AddLinearConstraint(
            Au,
            u_min - bu,
            u_max - bu,
            u_vars,
        ).evaluator().set_description("torque")


@jax.jit
def calculate_quadratic_cost(a: jax.Array, b: jax.Array):
    Q = 2 * jnp.diag(a * a)
    c = -2 * a * b
    return Q, c