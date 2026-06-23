"""Verify the vendored SO-101 URDF + meshes load (needed by the viser app)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("yourdfpy")

from lerobot_anyteleop.kinematics.urdf import load_urdf  # noqa: E402
from lerobot_anyteleop.robots import get_robot_spec  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_so101_urdf_loads_with_meshes():
    spec = get_robot_spec("so101")
    path = os.path.join(REPO, spec.urdf)
    assert os.path.exists(path), "run `anyteleop-fetch-urdf` to vendor the SO-101 URDF"
    if not os.path.isdir(os.path.join(os.path.dirname(path), "assets")):
        pytest.skip("SO-101 meshes not fetched; run `anyteleop-fetch-urdf`")
    urdf = load_urdf(path, load_meshes=True)
    # EE + base links present, and visual geometry actually loaded.
    assert spec.ee_link in urdf.link_map
    assert spec.base_link in urdf.link_map
    assert urdf.scene is not None and len(urdf.scene.geometry) > 0


def test_so101_urdf_loads_without_meshes_for_fk():
    spec = get_robot_spec("so101")
    path = os.path.join(REPO, spec.urdf)
    urdf = load_urdf(path, load_meshes=False)
    assert spec.ee_link in urdf.link_map
