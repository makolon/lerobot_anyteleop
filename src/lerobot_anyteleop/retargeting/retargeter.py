"""6-DOF end-effector retargeting from leader to follower.

Strategy (anchor / "clutch" based, drift-free):

* On :meth:`engage`, snapshot the leader EE pose (``L0``) and the follower EE
  pose (``F0``). ``F0`` becomes the origin the follower moves relative to.
* Each tick, measure the leader's motion *since the anchor* as a delta in the
  leader base frame::

      d_pos = p_L - p_L0
      d_rot = R_L * R_L0^-1

* Map that delta into the follower base frame via a fixed ``align`` rotation
  (handles the two arms being mounted in different orientations), scale position
  and orientation independently, then apply it on top of the follower anchor::

      p_F* = p_F0 + pos_scale * (R_align * d_pos)
      R_F* = scale_rot(R_align * d_rot * R_align^-1, ori_scale) * R_F0

A scale > 1 amplifies the operator's motion; < 1 gives fine control. Re-engaging
(clutch) re-anchors so the workspaces can be re-centered without moving the
follower — exactly like indexing a mouse.
"""

from __future__ import annotations

import numpy as np

from ..transforms import (
    Pose,
    quat_conjugate,
    quat_multiply,
    quat_rotate,
    scale_rotation,
)


class PoseRetargeter:
    def __init__(
        self,
        position_scale: float = 1.0,
        orientation_scale: float = 1.0,
        align_wxyz: np.ndarray | None = None,
    ) -> None:
        self.position_scale = float(position_scale)
        self.orientation_scale = float(orientation_scale)
        self._align = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if align_wxyz is None
            else np.asarray(align_wxyz, dtype=np.float64)
        )
        self._align_inv = quat_conjugate(self._align)

        self._engaged = False
        self._leader_anchor: Pose | None = None
        self._follower_anchor: Pose | None = None

    @property
    def is_engaged(self) -> bool:
        return self._engaged

    def engage(self, leader_pose: Pose, follower_pose: Pose) -> None:
        """Anchor the mapping at the current leader & follower poses."""
        self._leader_anchor = leader_pose.copy()
        self._follower_anchor = follower_pose.copy()
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False

    def compute_target(self, leader_pose: Pose) -> Pose:
        """Map the current leader pose to a follower EE target pose."""
        if not self._engaged or self._leader_anchor is None or self._follower_anchor is None:
            raise RuntimeError("PoseRetargeter.engage() must be called before compute_target().")

        # Leader delta in the leader base frame.
        d_pos = leader_pose.position - self._leader_anchor.position
        d_rot = quat_multiply(leader_pose.wxyz, quat_conjugate(self._leader_anchor.wxyz))

        # Re-express the delta in the follower base frame (similarity transform).
        d_pos_f = quat_rotate(self._align, d_pos)
        d_rot_f = quat_multiply(quat_multiply(self._align, d_rot), self._align_inv)

        # Scale position and orientation independently.
        d_pos_s = self.position_scale * d_pos_f
        d_rot_s = scale_rotation(d_rot_f, self.orientation_scale)

        # Apply on top of the follower anchor.
        target_pos = self._follower_anchor.position + d_pos_s
        target_quat = quat_multiply(d_rot_s, self._follower_anchor.wxyz)
        return Pose(position=target_pos, wxyz=target_quat)
