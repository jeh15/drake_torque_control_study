import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Custom imports:
from base_controller import BaseController

class QpWithDirConstraint(BaseController):
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

        # OSQP solver:
        self.solver, self.solver_options = make_osqp_solver_and_options()

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
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        num_v = len(v)
        M, C, tau_g = calc_dynamics(self.plant, self.context)
        Minv = inv(M)
        H = C - tau_g

        # Base QP formulation.
        Iv = np.eye(self.num_q)
        zv = np.zeros(self.num_q)
        prog = MathematicalProgram()

        X, V, Jt, Jtdot_v = pose_actual
        Mt, Mtinv, Jt, Jtbar, Nt_T = reproject_mass(Minv, Jt)
        Nt = Nt_T.T

        # Compute spatial feedback.
        kp_t, kd_t = self.gains.task
        num_t = 6
        It = np.eye(num_t)
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

        num_t = 6

        # # *very* sloppy looking
        # scale_A_t = np.eye(num_t)

        # # better, but may need relaxation
        # scale_A_t = np.ones((num_t, 1))

        # can seem "loose" towards end of traj for rotation
        # (small feedback -> scale a lot). relaxing only necessary for
        # implicit version.
        scale_A_t = np.array([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ]).T

        num_scales_t = scale_A_t.shape[1]
        scale_vars_t = prog.NewContinuousVariables(num_scales_t, "scale_t")
        desired_scales_t = np.ones(num_scales_t)

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
        # relax_primary = 1.0
        # relax_primary = 4.0
        # relax_primary = 1e1
        # relax_primary = 1e1
        # relax_primary = np.array([100, 100, 100, 50, 50, 50])
        # relax_primary = np.array([20, 20, 20, 10, 10, 10])
        # relax_primary = 50.0
        # relax_primary = 100.0  # OK
        # relax_primary = 200.0  # OK
        # relax_primary = 500.0  # maybe good?
        # relax_primary = 1e3
        # relax_primary = 1e4
        # relax_primary = 1e5
        # relax_primary = 1e6
        relax_secondary = None

        # norm_t = np.linalg.norm(edd_t_c)
        # min_t = 5.0
        # if norm_t < min_t:
        #     relax_primary = 1.0
        # print(norm_t)
        # assert num_scales_t == 1  # HACK
        # if norm_t < min_t:
        #     if self.prev_dir is not None:
        #         edd_t_c = min_t * self.prev_dir
        #         desired_scales_t[:] = norm_t / min_t
        # else:
        #     dir_t = edd_t_c / norm_t
        #     if self.should_save:
        #         self.prev_dir = dir_t

        kinematic = False

        if scale_secondary:
            scale_A_p = np.ones((num_v, 1))
            # scale_A_p = np.eye(num_v)
            num_scales_p = scale_A_p.shape[1]
            scale_vars_p = prog.NewContinuousVariables(num_scales_p, "scale_p")

        if relax_primary is not None:
            relax_vars_t = prog.NewContinuousVariables(num_t, "task.relax")
            relax_cost_t = np.ones(num_t) * relax_primary
            relax_proj_t = np.diag(relax_cost_t)
        if relax_secondary is not None:
            relax_vars_p = prog.NewContinuousVariables(num_v, "q relax")

        assert self.use_torque_weights
        proj_t = Jt.T @ Mt
        proj_p = Nt_T @ M
        # proj_p = Nt_T

        if implicit:
            vd_star = prog.NewContinuousVariables(self.num_q, "vd_star")
            u_star = prog.NewContinuousVariables(self.num_q, "u_star")

            # Dynamics constraint.
            dyn_vars = np.concatenate([vd_star, u_star])
            dyn_A = np.hstack([M, -Iv])
            dyn_b = -H
            prog.AddLinearEqualityConstraint(
                dyn_A, dyn_b, dyn_vars
            ).evaluator().set_description("dyn")

            u_vars = u_star
            Au = np.eye(num_v)
            bu = np.zeros(num_v)
            vd_vars = vd_star
            Avd = np.eye(num_v)
            bvd = np.zeros(num_v)
        else:
            # Primary, scale.
            u_vars = scale_vars_t
            Au = proj_t @ np.diag(edd_t_c) @ scale_A_t
            bu = -proj_t @ Jtdot_v + H

            if scale_secondary:
                Au_p = proj_p @ np.diag(edd_p_c) @ scale_A_p
                u_vars = np.concatenate([u_vars, scale_vars_p])
                Au = np.hstack([Au, Au_p])
            else:
                bu += proj_p @ edd_p_c

            if relax_primary is not None:
                Au_rt = proj_t
                u_vars = np.concatenate([u_vars, relax_vars_t])
                Au = np.hstack([Au, Au_rt])
                prog.Add2NormSquaredCost(
                    relax_proj_t,
                    np.zeros(num_t),
                    relax_vars_t,
                )
                # add_2norm_decoupled(
                #     prog,
                #     relax_cost_t,
                #     np.zeros(num_t),
                #     relax_vars_t,
                # )
            assert relax_secondary is None

            # Acceleration is just affine transform.
            vd_vars = u_vars
            Avd = Minv @ Au
            bvd = Minv @ (bu - H)

        # Add limits.
        vd_limits = self.plant_limits.vd
        # TODO(eric.cousineau): How to make this work correctly? Even
        # conservative estimate?
        torque_direct = True
        if torque_direct:
            u_vars = u_vars
        else:
            u_vars = None
            vd_tau_limits = vd_limits_from_tau(self.plant_limits.u, Minv, H)
            vd_limits = vd_limits.intersection(vd_tau_limits)
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

        add_manip_cbf = False

        if add_manip_cbf:
            # mu_min = 0.005
            mu_min = 0.001
            mu, Jmu = calc_manip_index(
                self.plant,
                self.context,
                self.frame_W,
                self.frame_G,
                self.plant_ad,
                self.context_ad,
            )
            if self.Jmu_prev is None:
                self.Jmu_prev = Jmu
            Jmudot = (Jmu - self.Jmu_prev) / dt
            self.Jmu_prev = Jmu
            Jmudot_v = Jmudot @ v
            amu_1 = lambda x: x
            amu_2 = amu_1
            kmu_1 = 2 / (dt * dt)
            kmu_2 = 2 / dt

            kmu_1 /= 5**2  # for high feedback gain
            # kmu_1 /= 10**2  # for low feedback gain
            kmu_2 /= 5

            h_mu = mu - mu_min
            hd_mu = Jmu @ v
            # hdd = J*vd + Jd_v >= -∑ Kᵢ h⁽ⁱ⁾
            Amu = Jmu @ Avd
            bmu = (
                -Jmudot_v
                -Jmu @ bvd
                -kmu_1 * amu_1(h_mu)
                -kmu_2 * amu_2(hd_mu)
            )
            prog.AddLinearConstraint(
                [Amu],
                [bmu],
                [np.inf],
                vd_vars,
            ).evaluator().set_description("manip cbf")

        # if implicit:
        #     # prog.AddBoundingBoxConstraint(
        #     #     self.plant_limits.u.lower,
        #     #     self.plant_limits.u.upper,
        #     #     u_star,
        #     # ).evaluator().set_description("u direct")
        #     prog.AddBoundingBoxConstraint(
        #         vd_tau_limits.lower,
        #         vd_tau_limits.upper,
        #         vd_star,
        #     ).evaluator().set_description("u via vd")
        # else:
        #     # u_min, u_max = self.plant_limits.u
        #     # prog.AddLinearConstraint(
        #     #     Au,
        #     #     u_min - bu,
        #     #     u_max - bu,
        #     #     vd_vars,
        #     # ).evaluator().set_description("u direct")
        #     vd_min, vd_max = vd_tau_limits
        #     prog.AddLinearConstraint(
        #         Avd,
        #         vd_min - bvd,
        #         vd_max - bvd,
        #         vd_vars,
        #     ).evaluator().set_description("u via vd")

        # TODO(eric.cousineau): Add CBF on manip index.

        if kinematic:
            Jtpinv = np.linalg.pinv(Jt)
            Nt_T = Iv - Jtpinv @ Jt

        # print(np.linalg.matrix_rank(Jt))
        # print(np.linalg.matrix_rank(Nt_T))

        # Constrain along desired tracking, J*vdot + Jdot*v = s*edd_c
        # For simplicity, allow each direction to have its own scaling.

        if implicit:
            task_vars_t = np.concatenate([vd_star, scale_vars_t])
            task_bias_t = edd_t_c
            task_A_t = np.hstack([Jt, -np.diag(task_bias_t) @ scale_A_t])
            task_b_t = -Jtdot_v

            if relax_primary is not None:
                task_vars_t = np.concatenate([task_vars_t, relax_vars_t])
                task_A_t = np.hstack([task_A_t, -It])
                # proj = proj_t
                if kinematic:
                    relax_proj_t = Jtpinv @ proj
                prog.Add2NormSquaredCost(
                    relax_proj_t @ It,
                    relax_proj_t @ np.zeros(num_t),
                    relax_vars_t,
                )

            prog.AddLinearEqualityConstraint(
                task_A_t, task_b_t, task_vars_t
            ).evaluator().set_description("task")

            if dup_eq_as_cost:
                prog.Add2NormSquaredCost(
                    dup_scale * task_A_t, dup_scale * task_b_t, task_vars_t
                )

        # Try to optimize towards scale=1.
        # proj = Jt.T @ Mt @ scale_A
        # proj = Mt @ scale_A
        # proj = scale_A
        # import pdb; pdb.set_trace()
        # proj *= 10
        # proj = proj * np.sqrt(num_scales)

        # proj = np.eye(num_scales_t)
        # prog.Add2NormSquaredCost(
        #     proj @ np.eye(num_scales_t),
        #     proj @ desired_scales_t,
        #     scale_vars_t,
        # )
        add_2norm_decoupled(
            prog,
            np.ones(num_scales_t),
            desired_scales_t,
            scale_vars_t,
        )

        # TODO(eric.cousineau): Maybe I need to constrain these error dynamics?

        # weight = self.posture_weight
        # task_proj = weight * task_proj
        # task_A = task_proj @ Iv
        # task_b = task_proj @ edd_c
        # prog.Add2NormSquaredCost(task_A, task_b, vd_star)

        if implicit:
            if not scale_secondary:
                assert not dup_eq_as_cost
                task_A_p = proj_p
                task_b_p = proj_p @ edd_p_c
                prog.AddLinearEqualityConstraint(
                    task_A_p, task_b_p, vd_star,
                ).evaluator().set_description("posture")
            else:
                task_bias_p = edd_p_c
                # task_bias_rep = np.tile(edd_c, (num_scales, 1)).T
                task_vars_p = np.concatenate([vd_star, scale_vars_p])
                task_A_p = np.hstack([Iv, -np.diag(task_bias_p) @ scale_A_p])
                task_b_p = np.zeros(num_v)
                # TODO(eric.cousineau): Weigh penalty based on how much feedback we
                # need?
                if relax_secondary is not None:
                    task_vars_p = np.concatenate([task_vars_p, relax_vars_p])
                    task_A_p = np.hstack([task_A_p, -Iv])
                    proj = proj_p
                    prog.Add2NormSquaredCost(
                        relax_secondary * proj @ Iv,
                        proj @ np.zeros(num_v),
                        relax_secondary,
                    )
                task_A_p = proj_p @ task_A_p
                task_b_p = proj_p @ task_b_p
                prog.AddLinearEqualityConstraint(
                    task_A_p, task_b_p, task_vars_p,
                ).evaluator().set_description("posture")
                if dup_eq_as_cost:
                    prog.Add2NormSquaredCost(
                        dup_scale * task_A_p, dup_scale * task_b_p, task_vars_p,
                    )

        if scale_secondary:
            desired_scales_p = np.ones(num_scales_p)
            # proj = self.posture_weight * np.eye(num_scales_p)
            # # proj = self.posture_weight * task_proj @ scale_A
            # # proj = self.posture_weight * scale_A
            # # proj = proj #/ np.sqrt(num_scales)
            # prog.Add2NormSquaredCost(
            #     proj @ np.eye(num_scales_p),
            #     proj @ desired_scales_p,
            #     scale_vars_p,
            # )
            add_2norm_decoupled(
                prog,
                self.posture_weight * np.ones(num_scales_p),
                self.posture_weight * desired_scales_p,
                scale_vars_p,
            )

        # ones = np.ones(num_scales_t)
        # prog.AddBoundingBoxConstraint(
        #     -10.0 * ones,
        #     10.0 * ones,
        #     scale_vars_t,
        # )
        # if scale_secondary:
        #     ones = np.ones(num_scales_p)
        #     prog.AddBoundingBoxConstraint(
        #         -10.0 * ones,
        #         10.0 * ones,
        #         scale_vars_p,
        #     )

        # print(f"edd_t_c: {edd_t_c}")
        # if scale_secondary:
        #     print(f"  edd_p_c: {edd_p_c}")

        # Solve.
        try:
            # TODO(eric.cousineau): OSQP does not currently accept
            # warm-starting:
            # https://github.com/RobotLocomotion/drake/blob/v1.15.0/solvers/osqp_solver.cc#L335-L336
            result = solve_or_die(
                self.solver, self.solver_options, prog, x0=self.prev_sol
            )
        except RuntimeError:
            # print(np.rad2deg(self.plant_limits.q.lower))
            # print(np.rad2deg(self.plant_limits.q.upper))
            # print(np.rad2deg(q))
            # print(self.plant_limits.v)
            # print(v)
            raise

        if implicit:
            # tol = 1e-3
            # tol = 1e-12  # snopt default
            # tol = 1e-10  # snopt, singular
            tol = 1e-6
            # tol = 1e-11  # osqp default
            # tol = 1e-3  # scs default
        else:
            # tol = 1e-14  # osqp, clp default
            # tol = 0  # gurobi default
            # tol = 1e-14  # snopt default
            # tol = 1e-10  # snopt, singular
            tol = 1e-8
            # tol = 1e-4
            # tol = 1e-14  # scs - primal infeasible, dual unbounded?
            # tol = 1e-11  # mosek default
            # tol = 1e-1  # HACK


        scale_t = result.GetSolution(scale_vars_t)

        # print(v)
        # print(f"scale t: {scale_t}")
        if relax_primary is not None:
            relax_t = result.GetSolution(relax_vars_t)
            # print(f"  relax: {relax_t}")
        else:
            relax_t = np.zeros(num_t)
        if scale_secondary:
            scale_p = result.GetSolution(scale_vars_p)
            # print(f"scale p: {scale_p}")
        if relax_secondary is not None:
            relax_p = result.GetSolution(relax_vars_p)
            # print(f"  relax: {relax_p}")
        # print("---")

        infeas = result.GetInfeasibleConstraintNames(prog, tol=tol)
        infeas_text = "\n" + indent("\n".join(infeas), "  ")
        assert len(infeas) == 0, infeas_text
        self.prev_sol = result.get_x_val()

        if implicit:
            tau = result.GetSolution(u_star)
        else:
            u_mul = scale_t
            if scale_secondary:
                u_mul = np.concatenate([u_mul, scale_p])
            if relax_primary is not None:
                u_mul = np.concatenate([u_mul, relax_t])
            tau = Au @ u_mul + bu

        tau = self.plant_limits.u.saturate(tau)

        # import pdb; pdb.set_trace()

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

        # sub = slice(3, 4)
        sub = slice(None, None)
        sel = (slice(None, None), sub)

        def plot_lim(limits):
            lower, upper = limits
            lower = lower[sub]
            upper = upper[sub]
            ts_lim = ts[[0, -1]]
            reset_color_cycle()
            plt.plot(ts_lim, [lower, lower], ":")
            reset_color_cycle()
            plt.plot(ts_lim, [upper, upper], ":")

        _, axs = plt.subplots(num=1, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, qs[sel])
        legend_for(qs[sel])
        plot_lim(self.plant_limits.q)
        plt.title("q")
        plt.sca(axs[1])
        plt.plot(ts, vs[sel])
        legend_for(vs[sel])
        plot_lim(self.plant_limits.v)
        plt.title("v")
        plt.sca(axs[2])
        plt.plot(ts, us[sel])
        legend_for(us[sel])
        plot_lim(self.plant_limits.u)
        plt.title("u")
        plt.tight_layout()

        # plt.show()
        # return  # HACK

        _, axs = plt.subplots(num=2, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, edd_ts)
        legend_for(edd_ts)
        plt.title("edd_t_c")
        plt.sca(axs[1])
        plt.plot(ts, edd_ps)
        legend_for(edd_ps)
        plt.title("edd_p_c")
        plt.sca(axs[2])
        plt.plot(ts, edd_ps_null)
        legend_for(edd_ps_null)
        plt.title("edd_p_c null")
        plt.tight_layout()

        _, axs = plt.subplots(num=3, nrows=3)
        plt.sca(axs[0])
        plt.plot(ts, s_ts)
        plt.ylim(-5, 5)
        legend_for(s_ts)
        plt.title("s_t")
        plt.sca(axs[1])
        if len(r_ts) > 0:
            plt.plot(ts, r_ts)
            legend_for(r_ts)
        plt.title("r_t")
        plt.sca(axs[2])
        if len(s_ps) > 0:
            plt.plot(ts, s_ps)
            plt.ylim(-5, 5)
            legend_for(s_ps)
        plt.title("s_p")
        plt.tight_layout()

        _, axs = plt.subplots(num=4, nrows=2)
        plt.sca(axs[0])
        plt.plot(ts, sigmas)
        legend_for(sigmas)
        plt.title("singular values")
        plt.sca(axs[1])
        manips = np.prod(sigmas, axis=-1)
        plt.plot(ts, manips)
        plt.title("manip index")
        plt.tight_layout()

        plt.show()
