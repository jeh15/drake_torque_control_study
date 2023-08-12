import functools
from typing import Tuple, List

import numpy as np
import jax
import jax.numpy as jnp

# Custom imports:
import controller_utilities

# type aliases:
Pair = Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]
Limits = List[Pair]
ConstraintOutput = Tuple[
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Tuple[jnp.ndarray, jnp.ndarray],
]

jax.config.update("jax_enable_x64", True)
dtype = jnp.float64


@functools.partial(
    jax.jit,
    static_argnames=[
        "dt", "q_scale", "v_scale"
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
    q_limits: Pair,
    v_limits: Pair,
    acceleration_limits: Pair,
    control_limits: Pair,
    task_subspace: jax.Array,
    postural_subspace: jax.Array,
    dt: float,
    q_scale: float = 20.0,
    v_scale: float = 10.0,
) -> ConstraintOutput:
    # Unpack Limits:
    q_min, q_max = q_limits
    v_min, v_max = v_limits
    vd_min, vd_max = acceleration_limits
    u_min, u_max = control_limits

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
    return (A, l, u), (A_combined, b)


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
    q_limits = (jnp.zeros((7,)), jnp.zeros((7,)))
    v_limits = (jnp.zeros((7,)), jnp.zeros((7,)))
    acceleration_limits = (jnp.zeros((7,)), jnp.zeros((7,)))
    control_limits = (jnp.zeros((7,)), jnp.zeros((7,)))

    # Run:
    (A, l, u), (A_control, b_control) = calculate_constraints(
        dummy_q,
        dummy_v,
        dummy_task_projection,
        dummy_postural_projection,
        desired_task_acceleration,
        desired_postural_acceleration,
        mass_matrix_inverse,
        bias_term,
        J_task_dv,
        q_limits,
        v_limits,
        acceleration_limits,
        control_limits,
        task_subsapce,
        postural_subspace,
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
