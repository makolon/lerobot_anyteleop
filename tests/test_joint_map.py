from __future__ import annotations

import numpy as np
import pytest

from lerobot_anyteleop.joint_utils import JointMap, reorder


def test_reorder_by_name_with_default():
    vals = {"b": 2.0, "a": 1.0}
    out = reorder(vals, ["a", "b", "c"], default=-1.0)
    assert np.allclose(out, [1.0, 2.0, -1.0])


def test_jointmap_identity():
    names = ["j1", "j2", "j3"]
    jm = JointMap(names, names)
    q = np.array([0.1, 0.2, 0.3])
    assert np.allclose(jm.to_arm(q), q)
    assert np.allclose(jm.to_full(q, base=np.zeros(3)), q)


def test_jointmap_reordered_and_extra():
    # full has an extra "finger" joint and a different order (mimics pyroki/panda).
    full = ["finger", "j2", "j1", "j3"]
    arm = ["j1", "j2", "j3"]
    jm = JointMap(full, arm)
    q_full = np.array([9.0, 2.0, 1.0, 3.0])  # finger=9
    assert np.allclose(jm.to_arm(q_full), [1.0, 2.0, 3.0])

    base = np.array([7.0, 0.0, 0.0, 0.0])  # finger base preserved
    out = jm.to_full([1.0, 2.0, 3.0], base=base)
    assert np.allclose(out, [7.0, 2.0, 1.0, 3.0])


def test_jointmap_missing_arm_joint_raises():
    with pytest.raises(ValueError):
        JointMap(["a", "b"], ["a", "z"])
