"""pyroki-based FK/IK model.

pyroki (https://github.com/chungmin99/pyroki) is a JAX toolkit for robot
kinematic optimization. Forward kinematics is a single call; inverse kinematics
is assembled from differentiable cost factors solved with ``jaxls``.

The IK problem here mirrors pyroki's reference ``solve_ik`` snippet
(``pose_cost_analytic_jac`` + ``limit_constraint``) and adds a small
``rest_cost`` toward the current configuration so consecutive teleop solutions
stay continuous (no elbow flips between ticks).
"""

from __future__ import annotations

import functools

import numpy as np

from ..transforms import Pose
from .base import KinematicsModel
from .urdf import load_urdf


class PyrokiKinematics(KinematicsModel):
    def __init__(
        self,
        urdf_source: str,
        ee_link_name: str,
        *,
        pos_weight: float = 50.0,
        ori_weight: float = 10.0,
        rest_weight: float = 0.1,
    ) -> None:
        # Imported here so the rest of the package (mock pipeline, tests) does not
        # require the JAX stack just to import.
        import jax.numpy as jnp
        import pyroki as pk

        self._jnp = jnp
        urdf = load_urdf(urdf_source)
        self._robot = pk.Robot.from_urdf(urdf)
        self._ee_link_name = ee_link_name

        link_names = list(self._robot.links.names)
        if ee_link_name not in link_names:
            raise ValueError(
                f"EE link {ee_link_name!r} not in URDF links. Available: {link_names}"
            )
        self._ee_index = link_names.index(ee_link_name)

        self._actuated_names = list(self._robot.joints.actuated_names)
        self._lower = np.asarray(self._robot.joints.lower_limits, dtype=np.float64)
        self._upper = np.asarray(self._robot.joints.upper_limits, dtype=np.float64)
        self._pos_weight = float(pos_weight)
        self._ori_weight = float(ori_weight)
        self._rest_weight = float(rest_weight)

    # -- metadata -----------------------------------------------------------
    @property
    def actuated_names(self) -> list[str]:
        return list(self._actuated_names)

    @property
    def num_actuated(self) -> int:
        return len(self._actuated_names)

    @property
    def lower_limits(self) -> np.ndarray:
        return self._lower.copy()

    @property
    def upper_limits(self) -> np.ndarray:
        return self._upper.copy()

    @property
    def ee_link_name(self) -> str:
        return self._ee_link_name

    # -- kinematics ---------------------------------------------------------
    def fk(self, q: np.ndarray) -> Pose:
        q = np.asarray(q, dtype=np.float64).reshape(self.num_actuated)
        poses = self._robot.forward_kinematics(self._jnp.asarray(q))  # (num_links, 7)
        wxyz_xyz = np.asarray(poses[self._ee_index])  # [qw,qx,qy,qz, x,y,z]
        return Pose(position=wxyz_xyz[4:], wxyz=wxyz_xyz[:4])

    def ik(self, target: Pose, q_init: np.ndarray | None = None) -> np.ndarray:
        rest = self.rest_pose() if q_init is None else np.asarray(q_init, dtype=np.float64)
        rest = rest.reshape(self.num_actuated)
        cfg = _solve_ik(
            self._robot,
            self._ee_link_name,
            np.asarray(target.wxyz, dtype=np.float64),
            np.asarray(target.position, dtype=np.float64),
            rest,
            self._pos_weight,
            self._ori_weight,
            self._rest_weight,
        )
        return np.asarray(cfg, dtype=np.float64)


# --------------------------------------------------------------------------- #
# IK solve (module-level so the jit cache is shared across model instances)
# --------------------------------------------------------------------------- #
def _solve_ik(
    robot,
    ee_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
    rest_pose: np.ndarray,
    pos_weight: float,
    ori_weight: float,
    rest_weight: float,
) -> np.ndarray:
    import jax.numpy as jnp

    target_link_index = list(robot.links.names).index(ee_link_name)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.asarray(target_wxyz),
        jnp.asarray(target_position),
        jnp.asarray(rest_pose),
        jnp.asarray(pos_weight),
        jnp.asarray(ori_weight),
        jnp.asarray(rest_weight),
    )
    return np.asarray(cfg)


@functools.lru_cache(maxsize=1)
def _jitted():
    """Lazily build & cache the jitted IK kernel (compiled on first call)."""
    import jax
    import jax_dataclasses as jdc
    import jaxlie
    import jaxls
    import pyroki as pk

    @jdc.jit
    def _kernel(robot, target_link_index, target_wxyz, target_position, rest_pose,
                pos_weight, ori_weight, rest_weight):
        joint_var = robot.joint_var_cls(0)
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_wxyz), target_position
        )
        costs = [
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                target_pose,
                target_link_index,
                pos_weight=pos_weight,
                ori_weight=ori_weight,
            ),
            pk.costs.limit_constraint(robot, joint_var),
            pk.costs.rest_cost(joint_var, rest_pose=rest_pose, weight=rest_weight),
        ]
        sol = (
            jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
            .analyze()
            .solve(
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            )
        )
        return sol[joint_var]

    return _kernel


def _solve_ik_jax(*args):
    return _jitted()(*args)
