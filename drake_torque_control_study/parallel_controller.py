import numpy as np

# Custom imports:
import controller_utilities
import geometry_utilities


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


def constraints():
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