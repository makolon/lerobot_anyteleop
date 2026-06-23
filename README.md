# lerobot_anyteleop

Kinematics-based teleoperation: drive a **follower robot** from a **LeRobot SO-101**
leader arm, recording multi-camera RealSense streams + robot state to **HDF5**
(convertible to LeRobot v2.1 / v3.0 datasets).

Followers are pluggable — **xArm7**, **Franka Panda / FR3**, and **UR5e** ship in
the box, and adding another arm is a registry entry + a small driver.

## How it works

The leader and follower have different kinematics, so motion is mapped in
**end-effector space**, not joint-by-joint:

```
SO-101 leader joints ──FK(pyroki)──▶ leader EE pose
                                          │
                              delta since clutch anchor
                                          │  × scale (position & orientation, 6-DOF)
                                          ▼
follower joints ◀──IK(pyroki)── follower EE target ◀── applied on the follower anchor
        │
        ├─▶ stream joint servo (xArm set_servo_angle_j / UR servoJ / Panda JointPosition)
        └─▶ record: joints + EE poses + RealSense frames ──▶ HDF5
```

* **FK/IK:** [pyroki](https://github.com/chungmin99/pyroki) (JAX). IK = analytic
  pose cost + joint-limit constraint + a small rest cost toward the current
  configuration (warm start) for smooth, jitter-free teleop. The kinematics is
  fully URDF-driven, so it's robot-agnostic.
* **Retargeting:** anchor/"clutch" based and drift-free. The follower moves
  relative to its pose at engage time; position and orientation scale
  independently. Re-engaging re-centers the workspaces (like indexing a mouse).
* **Joint mapping:** pyroki may expose more/reordered actuated joints than the
  hardware commands (e.g. a Panda finger joint). A name-based `JointMap` bridges
  the full FK/IK joint vector and the hardware arm joints, so any DOF works.

## Visualize it without hardware (viser)

The hardware-free path is an interactive 3D visualization that runs the **exact
same retargeting pipeline** as the real controller:

```bash
pixi run viz                                            # follower-only (xArm7)
pixi run -- anyteleop-viz --follower panda --no-leader  # or panda / ur5e
pixi run viz-with-leader                                # also render the SO-101 leader
# open the printed http://localhost:8080
```

Drag the leader joint sliders; the follower solves IK and moves in 3D, with live
position/orientation-scale sliders and a re-engage (clutch) button. By default
only the follower is shown (the SO-101 leader is just the input device); add
`viz-with-leader` / drop `--no-leader` to also render the leader.

![pipeline]: leader sliders → leader FK → retarget → follower IK → render

## Project layout

```
src/lerobot_anyteleop/
  transforms.py            # NumPy SE(3)/SO(3) math + Pose (no JAX)
  robots/registry.py       # RobotSpec registry: so101, xarm7, panda, ur5e
  joint_utils.py           # JointMap (arm <-> full), name-based reorder
  kinematics/              # pyroki FK/IK behind a KinematicsModel ABC
  retargeting/             # PoseRetargeter (6-DOF delta scaling)
  teleop/
    pipeline.py            # KinematicRetargetPipeline (FK->retarget->IK) — hardware-free core
    controller.py          # real-hardware control loop
  devices/
    leader/   {base, so101}                  # SO-101 via lerobot (lazy import)
    follower/ {base, xarm7, ur, franka}      # follower drivers, selected by backend
    camera/   {base, realsense, manager}     # RealSense D435 (lazy import)
  recording/hdf5_recorder.py # incremental, resizable, compressed HDF5
  config.py / factory.py     # YAML config -> kinematics/devices/system
  viz/viser_app.py           # interactive visualization
  cli/                       # anyteleop, -viz, -list-cameras, -fetch-urdf, -inspect, -convert
configs/ {xarm7,panda,ur5e}.yaml
assets/urdf/so101/           # vendored SO-101 URDF (meshes via anyteleop-fetch-urdf)
tests/
```

## Setup (pixi)

```bash
pixi install                 # kinematics + viser + tests; solves on linux-64 / osx-arm64
pixi run test
pixi run fetch-urdf          # download SO-101 meshes (needed for viser)
pixi run viz                 # visualize SO-101 -> xArm7

# real hardware — one environment per follower (each adds the SO-101 leader + RealSense):
pixi install -e xarm         #  xArm7  (xArm-Python-SDK)
pixi install -e ur           #  UR5e   (ur_rtde)
pixi install -e franka       #  Panda  (panda-python; needs RT kernel + FCI)

pixi run -e xarm anyteleop-list-cameras               # discover RealSense serials
# edit configs/xarm7.yaml (leader port, robot ip, serials), then:
pixi run -e xarm anyteleop --config configs/xarm7.yaml --record
```

> The `default` environment has the full **kinematics stack** (jax + pyroki +
> viser) so visualization and the whole retarget→IK pipeline are testable with no
> hardware. Robot/camera SDKs live only in the per-robot environments and are
> imported lazily.
>
> **RealSense on Apple Silicon:** the official `pyrealsense2` has no arm64 wheel;
> the `osx-arm64` target uses the community `pyrealsense2-macosx` build.

## Adding a follower robot

1. Add a `RobotSpec` to `robots/registry.py` (URDF source — a `robot_descriptions`
   name or a path — EE link, `arm_joint_names`, `home`, default `follower_backend`).
2. Add a driver in `devices/follower/` implementing `FollowerInterface` and
   register it in `devices/follower/__init__.py`.

Kinematics, retargeting, recording, and the viser app then work unchanged.

## Recorded HDF5 schema

One `episode_XXXXXX.hdf5` per episode (datasets grow per step):

| dataset | shape | dtype | meaning |
|---|---|---|---|
| `/observation/follower_qpos` | `(T, N)` | f32 | measured follower joints (rad) |
| `/observation/follower_ee_pose` | `(T,7)` | f32 | `[x,y,z, qw,qx,qy,qz]` |
| `/observation/leader_qpos` | `(T,6)` | f32 | 5 arm joints + gripper |
| `/observation/leader_ee_pose` | `(T,7)` | f32 | leader EE pose |
| `/observation/images/<cam>` | `(T,H,W,3)` | u8 | RGB (gzip, per-frame chunks) |
| `/observation/depth/<cam>` | `(T,H,W)` | u16 | optional |
| `/action/follower_qpos` | `(T, N)` | f32 | commanded joints (the action) |
| `/action/follower_ee_pose` | `(T,7)` | f32 | retargeted EE target |
| `/action/gripper` | `(T,1)` | f32 | normalized gripper |
| `/timestamp` | `(T,)` | f64 | seconds since episode start |

`N` = follower arm DOF (7 xArm7/Panda, 6 UR5e). Attributes store `fps`, `task`,
joint/camera names, `num_steps`.

```bash
anyteleop-inspect data/recordings/episode_000000.hdf5
```

## Convert to a LeRobot dataset

```bash
anyteleop-convert --input-dir data/recordings --dry-run     # inspect the mapping (no lerobot needed)
pixi run -e xarm anyteleop-convert --input-dir data/recordings --repo-id local/anyteleop
```

Mapping: `observation.state ← follower_qpos`, `action ← action/follower_qpos`,
`observation.images.<cam> ← images/<cam>`. The LeRobot dataset API changed across
v2.x/v3.0, so the write path is a version-flagged best-effort scaffold; the
`--dry-run` mapping is stable. See `cli/convert_to_lerobot.py`.

## Hardware notes / things to verify on a real rig

* **Leader units:** SO-101 `get_action()` returns degrees (calibration-centered)
  for the 5 arm joints + `0..100` for the gripper. If the leader calibration zero
  differs from the URDF zero, set `leader.joint_sign` / `joint_offset`.
* **Servo modes:** xArm streams `set_servo_angle_j` in mode 1; UR uses `servoJ`
  (set `follower.options.servo_dt` ≈ `1/rate_hz`); Panda uses a 1 kHz
  `JointPosition` controller (raise `loop.rate_hz`). Keep per-step deltas small.
* **Frame alignment:** if leader and follower are mounted differently, set
  `retarget.align_rpy` so leader deltas map sensibly onto the follower.
* **Workspace:** IK clamps to joint limits; tune `retarget.position_scale` so the
  leader workspace maps inside the follower's reach.
