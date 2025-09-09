# pyright: reportPrivateUsage=false

import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from mocap_converter.io.bvh.channel_layout import BVHChannelLayout
from mocap_converter.io.bvh.saver import _build_hierarchy_string, save_bvh
from mocap_converter.kinematic_tree import KinematicTree
from mocap_converter.io.bvh.loader import load_bvh
from mocap_converter.motion_data import MotionData


def test_build_hierarchy_string() -> None:
    kinematic_tree: KinematicTree = KinematicTree.from_params(
        [
            {"name": "root", "parent_name": None},
            {"name": "child1", "parent_name": "root"},
            {"name": "child2", "parent_name": "root"},
            {"name": "child3", "parent_name": "child1"},
            {"name": "child4", "parent_name": "child1"},
            {"name": "child5", "parent_name": "child2"},
        ]
    )

    # Provide explicit channel layouts: root has positions; others rotations only
    channel_layouts: dict[str, BVHChannelLayout] = {
        name: BVHChannelLayout.from_rotation_order("ZXY", has_position_channels=(name == "root"))
        for name in ("root", "child1", "child2", "child3", "child4", "child5")
    }
    hierarchy, _ = _build_hierarchy_string(kinematic_tree, channel_layouts)

    expected_hierarchy = """
HIERARCHY
ROOT root
{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT child1
  {
    OFFSET 0.000000 0.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT child3
    {
      OFFSET 0.000000 0.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.000000 0.000000 0.000000
      }
    }
    JOINT child4
    {
      OFFSET 0.000000 0.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.000000 0.000000 0.000000
      }
    }
  }
  JOINT child2
  {
    OFFSET 0.000000 0.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.000000 0.000000 0.000000
    }
  }
}
""".strip()

    assert hierarchy.strip() == expected_hierarchy


def test_save_bvh_motion_values() -> None:
    # Build simple tree: root with two leaf children (siblings)
    kinematic_tree: KinematicTree = KinematicTree.from_params(
        [
            {"name": "root", "parent_name": None},
            {"name": "child1", "parent_name": "root"},
            {"name": "child2", "parent_name": "root"},
        ]
    )

    # Two frames of positions for root
    root_pos: NDArray[np.float64] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)

    # Define Euler angles (ZXY order) for each node, two frames
    root_euler: NDArray[np.float64] = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)  # Z, X, Y
    child1_euler: NDArray[np.float64] = np.array([[15.0, 5.0, 25.0], [35.0, 10.0, 45.0]], dtype=np.float64)
    child2_euler: NDArray[np.float64] = np.array([[5.0, 15.0, 10.0], [25.0, 35.0, 20.0]], dtype=np.float64)

    # Convert to quaternions for MotionData
    root_quat: NDArray[np.float64] = R.from_euler("ZXY", root_euler, degrees=True).as_quat()
    child1_quat: NDArray[np.float64] = R.from_euler("ZXY", child1_euler, degrees=True).as_quat()
    child2_quat: NDArray[np.float64] = R.from_euler("ZXY", child2_euler, degrees=True).as_quat()

    positions: dict[str, NDArray[np.float64]] = {"root": root_pos}
    rotations: dict[str, NDArray[np.float64]] = {
        "root": root_quat,
        "child1": child1_quat,
        "child2": child2_quat,
    }

    motion = MotionData(kinematic_tree, positions=positions, rotations=rotations, frame_time=1 / 30)

    # Explicit channel layouts
    channel_layouts: dict[str, BVHChannelLayout] = {
        "root": BVHChannelLayout.from_rotation_order("ZXY", has_position_channels=True),
        "child1": BVHChannelLayout.from_rotation_order("ZXY", has_position_channels=False),
        "child2": BVHChannelLayout.from_rotation_order("ZXY", has_position_channels=False),
    }

    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        out_path: Path = Path(f.name)

    try:
        save_bvh(motion, str(out_path), channel_layouts)
        text: str = out_path.read_text()

        # Locate motion section and parse frame lines
        lines: list[str] = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        motion_idx: int = lines.index("MOTION")
        assert lines[motion_idx + 1].startswith("Frames: 2")
        assert lines[motion_idx + 2].startswith("Frame Time: 0.033333")

        frame_lines: list[str] = lines[motion_idx + 3 :]
        assert len(frame_lines) == 2

        parsed: NDArray[np.float64] = np.array([[float(x) for x in ln.split()] for ln in frame_lines], dtype=np.float64)

        # Expected concatenation order: root pos(3) + root euler(3) + child1 euler(3) + child2 euler(3)
        expected_root_euler: NDArray[np.float64] = R.from_quat(root_quat).as_euler("ZXY", degrees=True)
        expected_child1_euler: NDArray[np.float64] = R.from_quat(child1_quat).as_euler("ZXY", degrees=True)
        expected_child2_euler: NDArray[np.float64] = R.from_quat(child2_quat).as_euler("ZXY", degrees=True)
        expected: NDArray[np.float64] = np.concatenate(
            [root_pos, expected_root_euler, expected_child1_euler, expected_child2_euler], axis=1
        )

        assert parsed.shape == expected.shape
        assert np.allclose(parsed, expected, atol=1e-6)
    finally:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass


def test_save_load_roundtrip_with_fixture() -> None:
    # Locate fixture BVH
    tests_dir: Path = Path(__file__).resolve().parents[2]
    fixture_path: Path = tests_dir / "fixtures" / "calibration1.bvh"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    # Load original BVH
    original: MotionData = load_bvh(str(fixture_path))

    # Save to a temp file with defaults (root pos + default rotation order)
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        out_path: Path = Path(f.name)

    try:
        save_bvh(original, str(out_path))
        reloaded: MotionData = load_bvh(str(out_path))

        # Compare frame meta
        assert reloaded.frame_count == original.frame_count
        assert abs(reloaded.frame_time - original.frame_time) < 1e-9

        # Compare positions and rotations per node present in original MotionData
        for node_name in original.kinematic_tree.nodes.keys():
            if original.has_positions(node_name):
                pos_orig: NDArray[np.float64] = original.positions[node_name]
                pos_new: NDArray[np.float64] = reloaded.positions[node_name]
                assert pos_orig.shape == pos_new.shape
                assert np.allclose(pos_orig, pos_new, atol=1e-6)

            if original.has_rotations(node_name):
                rot_orig: NDArray[np.float64] = original.rotations[node_name]
                rot_new: NDArray[np.float64] = reloaded.rotations[node_name]
                assert rot_orig.shape == rot_new.shape
                assert np.allclose(rot_orig, rot_new, atol=1e-6)
    finally:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
