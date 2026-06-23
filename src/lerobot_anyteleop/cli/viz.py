"""``anyteleop-viz`` — interactive viser visualization of the retargeting."""

from __future__ import annotations

import argparse

from ..viz import run_viser


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--follower", default="xarm7", help="xarm7 | panda | ur5e")
    p.add_argument("--leader", default="so101")
    p.add_argument("--position-scale", type=float, default=1.5)
    p.add_argument("--orientation-scale", type=float, default=1.0)
    p.add_argument("--offset", type=float, default=0.8, help="Follower base x-offset (m).")
    p.add_argument("--no-leader", action="store_true",
                   help="Hide the SO-101 leader robot; show only the follower (centered).")
    p.add_argument("--gripper-model", default=None,
                   help="none | xarm | robotiq_2f85 | franka | <urdf path/robot_descriptions name>. "
                        "Default: per-follower (xarm7->xarm, ur5e->robotiq_2f85, panda->franka).")
    p.add_argument("--gripper-mount", type=float, nargs=6, default=None,
                   metavar=("X", "Y", "Z", "R", "P", "Y"),
                   help="Flange->gripper mount offset (m + rad) for mounted grippers.")
    p.add_argument("--leader-port", default=None,
                   help="Serial port of the REAL SO-101 leader (e.g. /dev/ttyACM0). "
                        "If omitted, the leader is driven by GUI sliders.")
    p.add_argument("--leader-id", default="so101_leader", help="SO-101 calibration id.")
    p.add_argument("--calibrate", action="store_true",
                   help="Run the SO-101 calibration routine on connect.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args(argv)

    mount = args.gripper_mount
    run_viser(
        follower_robot=args.follower,
        leader_robot=args.leader,
        position_scale=args.position_scale,
        orientation_scale=args.orientation_scale,
        follower_offset=args.offset,
        show_leader=not args.no_leader,
        gripper_model=args.gripper_model,
        gripper_mount_xyz=tuple(mount[:3]) if mount else None,
        gripper_mount_rpy=tuple(mount[3:]) if mount else None,
        leader_port=args.leader_port,
        leader_id=args.leader_id,
        leader_calibrate=args.calibrate,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
