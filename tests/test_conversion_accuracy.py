import os
from typing import Dict, List
import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from bvh_converter.io.bvh.loader import load_bvh
from bvh_converter.pos2rot import get_rotations_from_positions
from bvh_converter.rot2pos import get_positions_from_rotations
from bvh_converter.motion_data import MotionData


@pytest.fixture
def sample_bvh_path() -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures", "calibration1.bvh")


@pytest.fixture
def motion_data(sample_bvh_path: str) -> MotionData:
    return load_bvh(sample_bvh_path)


def test_round_trip_conversion(motion_data: MotionData):
    """
    Test the round-trip conversion from rotations to positions and back to rotations.
    This ensures that the conversion maintains the integrity of the motion data.
    """

    print("Testing round-trip conversion...")

    # Get the root node and its initial position
    root_node = motion_data.kinematic_tree.root
    assert root_node is not None, "Root node should exist"

    root_pos: np.ndarray = motion_data.get_positions(root_node.name)

    # Convert rotations to positions
    converted_positions: Dict[str, np.ndarray] = get_positions_from_rotations(
        motion_data, root_pos, root_node, R.identity(), scale=1
    )

    positional_data: MotionData = MotionData(
        motion_data.kinematic_tree, positions=converted_positions, rotations=None, frame_time=motion_data.frame_time
    )

    # Convert back to rotations
    converted_rotations: Dict[str, R] = get_rotations_from_positions(positional_data, root_node)

    # Compare with original rotations
    for node_name, converted_rotation in converted_rotations.items():
        if motion_data.has_rotations(node_name):
            original_rotation: R = R.from_quat(motion_data.get_rotations(node_name))

            # Calculate relative rotation
            relative_rots: R = converted_rotation * original_rotation.inv()

            # Calculate angular error
            angle_errors = relative_rots.magnitude()
            mean_angle_error = np.mean(angle_errors)
            max_angle_error = np.max(angle_errors)

            print(
                f"{node_name}: Mean angle error: {mean_angle_error:.4f} rad, Max angle error: {max_angle_error:.4f} rad"
            )

            # Allow for reasonable numerical error
            assert mean_angle_error < 0.1, f"Angular error for {node_name} too large: {mean_angle_error} rad"
