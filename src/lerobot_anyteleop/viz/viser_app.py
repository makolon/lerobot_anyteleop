"""Interactive viser visualization of leader -> follower kinematic retargeting.

Drive the SO-101 leader with GUI joint sliders; the leader's end-effector pose is
forward-kinematically computed, retargeted (with live position/orientation scale
sliders), and solved with IK on the chosen follower (xArm7 / Panda / UR5e). Both
robots render in 3D, along with the follower's EE target frame.

This is the hardware-free way to develop, demo and validate the retargeting. It
reuses the exact same :class:`KinematicRetargetPipeline` as the real controller.

Run::

    anyteleop-viz --follower xarm7      # then open the printed http://localhost:8080
"""

from __future__ import annotations

import os
import time

import numpy as np

from ..config import FollowerConfig, LeaderConfig, RetargetConfig
from ..factory import build_pipeline
from ..joint_utils import reorder
from ..kinematics.urdf import load_urdf
from ..robots import get_robot_spec


def _slider_limits(kin, name: str) -> tuple[float, float]:
    names = kin.actuated_names
    if name in names:
        i = names.index(name)
        lo, hi = float(kin.lower_limits[i]), float(kin.upper_limits[i])
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            return lo, hi
    return -np.pi, np.pi


def run_viser(
    follower_robot: str = "xarm7",
    leader_robot: str = "so101",
    *,
    position_scale: float = 1.5,
    orientation_scale: float = 1.0,
    follower_offset: float = 0.8,
    host: str = "0.0.0.0",
    port: int = 8080,
    rate_hz: float = 30.0,
) -> None:
    import viser
    from viser.extras import ViserUrdf

    leader_cfg = LeaderConfig(robot=leader_robot)
    follower_cfg = FollowerConfig(robot=follower_robot)
    retarget = RetargetConfig(position_scale=position_scale, orientation_scale=orientation_scale)
    pipeline, leader_kin, follower_kin, leader_spec, follower_spec = build_pipeline(
        leader_cfg, follower_cfg, retarget
    )
    follower_home = np.asarray(follower_spec.home, dtype=np.float64)

    # Follower URDFs (xArm7 / Panda / UR5e) are fetched + cached automatically by
    # robot_descriptions on first use. Only the SO-101 leader meshes are vendored.
    leader_urdf_path = leader_spec.urdf
    if leader_urdf_path.endswith(".urdf"):
        mesh_dir = os.path.join(os.path.dirname(leader_urdf_path), "assets")
        if not os.path.isdir(mesh_dir):
            print(f"[viz] note: SO-101 meshes not found at {mesh_dir} — the leader will "
                  f"render without geometry. Run `anyteleop-fetch-urdf` to fetch them.")

    server = viser.ViserServer(host=host, port=port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    # Leader at origin, follower offset along +x so they don't overlap.
    server.scene.add_frame("/leader_base", show_axes=False)
    server.scene.add_frame(
        "/follower_base", position=(follower_offset, 0.0, 0.0), wxyz=(1.0, 0.0, 0.0, 0.0),
        show_axes=False,
    )
    leader_vis = ViserUrdf(
        server, load_urdf(leader_spec.urdf, load_meshes=True), root_node_name="/leader_base"
    )
    follower_vis = ViserUrdf(
        server, load_urdf(follower_spec.urdf, load_meshes=True), root_node_name="/follower_base"
    )
    leader_vis_names = list(leader_vis.get_actuated_joint_names())
    follower_vis_names = list(follower_vis.get_actuated_joint_names())

    # --- GUI ---------------------------------------------------------------
    server.gui.add_markdown(f"### Leader **{leader_robot}** -> Follower **{follower_robot}**")
    joint_sliders = {}
    for name in leader_spec.arm_joint_names:
        lo, hi = _slider_limits(leader_kin, name)
        joint_sliders[name] = server.gui.add_slider(
            name, min=lo, max=hi, step=1e-3, initial_value=0.0
        )
    gripper_slider = server.gui.add_slider("gripper", min=0.0, max=1.0, step=1e-2, initial_value=0.5)

    pos_scale_gui = server.gui.add_slider("pos scale", 0.1, 4.0, 0.1, position_scale)
    ori_scale_gui = server.gui.add_slider("ori scale", 0.0, 2.0, 0.1, orientation_scale)
    reengage_btn = server.gui.add_button("re-engage (clutch)")
    status = server.gui.add_markdown("")

    def leader_joint_dict() -> dict[str, float]:
        return {n: s.value for n, s in joint_sliders.items()}

    # --- initial engage ----------------------------------------------------
    state = {"q_arm": follower_home.copy(), "reengage": True}

    @reengage_btn.on_click
    def _(_) -> None:
        state["reengage"] = True

    leader_vis.update_cfg(reorder(leader_joint_dict(), leader_vis_names))
    follower_vis.update_cfg(
        reorder(pipeline.jmap.full_dict(pipeline.jmap.to_full(follower_home, follower_kin.rest_pose())),
                follower_vis_names)
    )
    target_frame = server.scene.add_frame("/follower_base/ee_target", axes_length=0.1, axes_radius=0.004)

    print(f"[viz] serving at http://localhost:{port}  (leader={leader_robot}, follower={follower_robot})")

    period = 1.0 / rate_hz if rate_hz > 0 else 0.0
    while True:
        ljd = leader_joint_dict()
        pipeline.retargeter.position_scale = pos_scale_gui.value
        pipeline.retargeter.orientation_scale = ori_scale_gui.value

        if state["reengage"]:
            pipeline.engage(ljd, state["q_arm"])
            state["reengage"] = False

        out = pipeline.step(ljd, state["q_arm"])
        state["q_arm"] = out.follower_q_arm  # perfect (kinematic) tracking

        # update robots
        leader_vis.update_cfg(reorder(ljd, leader_vis_names))
        follower_vis.update_cfg(
            reorder(pipeline.jmap.full_dict(out.follower_q_full), follower_vis_names)
        )
        # update target frame (expressed in follower base frame)
        tp = out.follower_target_pose
        target_frame.position = tuple(tp.position)
        target_frame.wxyz = tuple(tp.wxyz)

        lp = out.leader_pose.position
        status.content = (
            f"leader EE: [{lp[0]:.3f}, {lp[1]:.3f}, {lp[2]:.3f}]  "
            f"follower target: [{tp.position[0]:.3f}, {tp.position[1]:.3f}, {tp.position[2]:.3f}]"
        )
        if period:
            time.sleep(period)
