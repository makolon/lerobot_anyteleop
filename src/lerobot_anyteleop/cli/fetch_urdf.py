"""``anyteleop-fetch-urdf`` — download the SO-101 URDF + meshes into ``assets/``.

The ``.urdf`` is committed; the STL meshes (~15 MB) are fetched on demand (needed
only for the viser visualization — FK/IK does not load meshes). Follower URDFs
(xArm7 / Panda / UR5e) are fetched automatically by ``robot_descriptions``.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101"
URDF_NAME = "so101_new_calib.urdf"
MESHES = [
    "base_motor_holder_so101_v1.stl", "base_so101_v2.stl", "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl", "moving_jaw_so101_v1.stl", "rotation_pitch_so101_v1.stl",
    "sts3215_03a_no_horn_v1.stl", "sts3215_03a_v1.stl", "under_arm_so101_v1.stl",
    "upper_arm_so101_v1.stl", "waveshare_mounting_plate_so101_v2.stl",
    "wrist_roll_follower_so101_v1.stl", "wrist_roll_pitch_so101_v2.stl",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", default="assets/urdf/so101", help="Destination directory.")
    p.add_argument("--no-meshes", action="store_true", help="URDF only (enough for FK/IK).")
    args = p.parse_args(argv)

    dest = Path(args.dest)
    (dest / "assets").mkdir(parents=True, exist_ok=True)

    print(f"URDF  -> {dest / URDF_NAME}")
    urllib.request.urlretrieve(f"{BASE}/{URDF_NAME}", dest / URDF_NAME)

    if not args.no_meshes:
        for name in MESHES:
            out = dest / "assets" / name
            urllib.request.urlretrieve(f"{BASE}/assets/{name}", out)
            print(f"mesh  -> {out} ({out.stat().st_size} bytes)")
    print("Done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
