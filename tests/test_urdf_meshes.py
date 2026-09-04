"""Verify the vendored SO-101 URDF + meshes load (needed by the viser app)."""

from __future__ import annotations

import os

import numpy as np
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


def test_crx10ia_l_kinematic_urdf_has_exact_l_chain_and_limits():
    spec = get_robot_spec("crx10ia_l")
    path = os.path.join(REPO, spec.urdf)
    urdf = load_urdf(path, load_meshes=False)
    assert spec.ee_link in urdf.link_map
    assert tuple(urdf.actuated_joint_names) == spec.arm_joint_names

    # Lock all values copied from ws_fanuc's flattened CRX-10iA/L xacro. This
    # catches accidental substitution of the shorter CRX-10iA model as well as
    # axis-sign and flange-frame regressions that a visual check would miss.
    expected = {
        "J1": (
            (0, 0, 0.245),
            (0, 0, 1),
            -3.139847324337799,
            3.139847324337799,
            2.0943951023931953,
        ),
        "J2": (
            (0, 0, 0),
            (0, 1, 0),
            -3.139847324337799,
            3.139847324337799,
            2.0943951023931953,
        ),
        "J3": (
            (0, 0, 0.710),
            (0, -1, 0),
            -4.71238898038469,
            4.71238898038469,
            3.141592653589793,
        ),
        "J4": (
            (0, 0, 0),
            (-1, 0, 0),
            -3.3161255787892263,
            3.3161255787892263,
            3.141592653589793,
        ),
        "J5": (
            (0.540, 0, 0),
            (0, -1, 0),
            -3.139847324337799,
            3.139847324337799,
            3.141592653589793,
        ),
        "J6": (
            (0, -0.150, 0),
            (-1, 0, 0),
            -3.9269908169872414,
            3.9269908169872414,
            3.141592653589793,
        ),
    }
    for name, (xyz, axis, lower, upper, velocity) in expected.items():
        joint = urdf.joint_map[name]
        np.testing.assert_allclose(joint.origin[:3, 3], xyz, atol=1e-12)
        np.testing.assert_allclose(joint.axis, axis, atol=1e-12)
        assert joint.limit.lower == pytest.approx(lower)
        assert joint.limit.upper == pytest.approx(upper)
        assert joint.limit.velocity == pytest.approx(velocity)

    np.testing.assert_allclose(
        urdf.joint_map["J6-flange"].origin[:3, 3], (0.160, 0, 0), atol=1e-12
    )
    np.testing.assert_allclose(
        urdf.joint_map["flange-fanuc_flange"].origin,
        np.array(
            [
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        ),
        atol=1e-12,
    )
