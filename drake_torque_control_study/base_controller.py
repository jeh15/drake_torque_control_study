from pydrake.systems.framework import LeafSystem

from drake_torque_control_study.limits import PlantLimits
from drake_torque_control_study.systems import declare_simple_init

import controller_utilities


class BaseController(LeafSystem):
    def __init__(self, plant, frame_W, frame_G):
        super().__init__()
        self.plant = plant
        self.frame_W = frame_W
        self.frame_G = frame_G
        self.context = plant.CreateDefaultContext()
        self.num_q = plant.num_positions()
        self.num_x = 2 * self.num_q
        assert plant.num_velocities() == self.num_q
        self.state_input = self.DeclareVectorInputPort("state", self.num_x)
        self.torques_output = self.DeclareVectorOutputPort(
            "torques",
            size=self.num_q,
            calc=self.calc_torques,
        )
        self.get_init_state = declare_simple_init(
            self,
            self.on_init,
        )
        self.check_limits = True
        self.nominal_limits = PlantLimits.from_plant(plant)
        # Will be set externally.
        self.traj = None

    def on_init(self, sys_context, init):
        x = self.state_input.Eval(sys_context)
        self.plant.SetPositionsAndVelocities(self.context, x)
        q = self.plant.GetPositions(self.context)
        init.q = q

    def calc_torques(self, sys_context, output):
        x = self.state_input.Eval(sys_context)
        t = sys_context.get_time()

        tol = 1e-4
        self.plant.SetPositionsAndVelocities(self.context, x)
        if self.check_limits:
            q = self.plant.GetPositions(self.context)
            v = self.plant.GetVelocities(self.context)
            self.nominal_limits.assert_values_within_limits(q=q, v=v, tol=tol)

        init = self.get_init_state(sys_context)
        q0 = init.q
        pose_actual = controller_utilities.calc_spatial_values(
            self.plant, self.context, self.frame_W, self.frame_G
        )
        pose_desired = self.traj(t)
        tau = self.calc_control(t, pose_actual, pose_desired, q0)

        if self.check_limits:
            self.nominal_limits.assert_values_within_limits(u=tau, tol=tol)

        output.set_value(tau)

    def calc_control(self, t, pose_actual, pose_desired, q0):
        raise NotImplementedError()

    def show_plots(self):
        pass
