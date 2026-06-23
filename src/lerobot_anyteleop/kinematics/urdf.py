"""URDF loading helpers (thin wrapper over yourdfpy / robot_descriptions)."""

from __future__ import annotations

import os


def load_urdf(source: str, *, load_meshes: bool = False):
    """Load a URDF as a ``yourdfpy.URDF``.

    ``source`` is either

    * a ``robot_descriptions`` module name (e.g. ``"xarm7_description"``), which is
      downloaded/cached on demand, or
    * a path to a ``.urdf`` file on disk.

    Meshes are skipped by default: forward/inverse kinematics only need the
    kinematic tree (joint origins + axes), not the visual/collision geometry, so
    we avoid the cost and the missing-mesh warnings.
    """
    import yourdfpy

    looks_like_path = source.endswith(".urdf") or os.path.sep in source or os.path.exists(source)
    if not looks_like_path:
        # robot_descriptions module name.
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        return load_robot_description(source, load_meshes=load_meshes)

    if not os.path.exists(source):
        raise FileNotFoundError(f"URDF not found: {source!r}")

    return yourdfpy.URDF.load(
        source,
        load_meshes=load_meshes,
        build_scene_graph=True,
        load_collision_meshes=False,
        build_collision_scene_graph=False,
    )
