import os
from typing import Dict, List
import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from bvh_converter.io.bvh.loader import load_bvh
from bvh_converter.io.bvh.saver import BVHSaver
from bvh_converter.pos2rot import get_rotation_from_position
from bvh_converter.rot2pos import get_positions_from_rotations
from bvh_converter.motion_data import MotionData


@pytest.fixture
def sample_bvh_path() -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures", "calibration1.bvh")


@pytest.fixture
def motion_data(sample_bvh_path: str) -> MotionData:
    return load_bvh(sample_bvh_path)


def create_single_frame_motion_data(motion_data: MotionData, frame_idx: int) -> MotionData:
    """Helper function to create single-frame motion data."""
    single_frame_positions: Dict[str, np.ndarray] = {}
    single_frame_rotations: Dict[str, np.ndarray] = {}
    
    for node_name in motion_data.kinematic_tree.nodes.keys():
        if motion_data.has_positions(node_name):
            single_position: np.ndarray = motion_data.get_positions(node_name)[frame_idx:frame_idx+1]
            single_frame_positions[node_name] = single_position
        
        if motion_data.has_rotations(node_name):
            single_rotation: np.ndarray = motion_data.get_rotations(node_name)[frame_idx:frame_idx+1]
            single_frame_rotations[node_name] = single_rotation
    
    return MotionData(
        motion_data.kinematic_tree,
        positions=single_frame_positions if single_frame_positions else None,
        rotations=single_frame_rotations if single_frame_rotations else None,
        frame_time=motion_data.frame_time
    )


def test_rotational_to_positional_conversion_sample(motion_data: MotionData) -> None:
    """Test conversion from rotational to positional motion data using a few sample frames."""
    root_node = motion_data.kinematic_tree.root
    assert root_node is not None, "Root node should exist"
    
    # Test only a few frames for performance
    test_frames: List[int] = [0, motion_data.frame_count // 2, motion_data.frame_count - 1]
    
    for frame_idx in test_frames:
        # Create single-frame motion data
        single_frame_data: MotionData = create_single_frame_motion_data(motion_data, frame_idx)
        
        # Get root position for this frame
        if motion_data.has_positions(root_node.name):
            root_pos: np.ndarray = motion_data.get_positions(root_node.name)[frame_idx]
        else:
            root_pos = np.zeros(3)
        
        # Convert rotations to positions for this frame
        frame_positions: Dict[str, np.ndarray] = get_positions_from_rotations(
            single_frame_data, root_pos, root_node, R.identity(), scale=1
        )
        
        # Compare with original positions (if available)
        for node_name, converted_position in frame_positions.items():
            if motion_data.has_positions(node_name):
                original_position: np.ndarray = motion_data.get_positions(node_name)[frame_idx]
                
                # Handle both single positions and multi-frame positions
                if isinstance(converted_position, np.ndarray) and converted_position.ndim == 2:
                    converted_pos: np.ndarray = converted_position[0]
                else:
                    converted_pos = converted_position
                
                # Calculate error
                error = float(np.linalg.norm(original_position - converted_pos))
                print(f"Frame {frame_idx}, {node_name}: position error = {error:.6f}")
                
                assert error < 1e-6, f"Position error for {node_name} at frame {frame_idx} too large: {error}"


def test_positional_to_rotational_conversion_sample(motion_data: MotionData) -> None:
    """Test conversion from positional to rotational motion data using a few sample frames."""
    root_node = motion_data.kinematic_tree.root
    assert root_node is not None, "Root node should exist"
    
    # Test only a few frames for performance
    test_frames: List[int] = [0, motion_data.frame_count // 2, motion_data.frame_count - 1]
    
    for frame_idx in test_frames:
        # First create positional data from rotational data
        single_frame_rot_data: MotionData = create_single_frame_motion_data(motion_data, frame_idx)
        
        if motion_data.has_positions(root_node.name):
            root_pos: np.ndarray = motion_data.get_positions(root_node.name)[frame_idx]
        else:
            root_pos = np.zeros(3)
        
        frame_positions: Dict[str, np.ndarray] = get_positions_from_rotations(
            single_frame_rot_data, root_pos, root_node, R.identity(), scale=1
        )
        
        # Create positional motion data
        single_frame_positions: Dict[str, np.ndarray] = {}
        for node_name, position in frame_positions.items():
            if isinstance(position, np.ndarray) and position.ndim == 2:
                single_frame_positions[node_name] = position
            else:
                single_frame_positions[node_name] = position.reshape(1, -1)
        
        positional_data: MotionData = MotionData(
            motion_data.kinematic_tree,
            positions=single_frame_positions,
            rotations=None,
            frame_time=motion_data.frame_time
        )
        
        # Convert back to rotations
        frame_rotations: Dict[str, R] = get_rotation_from_position(
            positional_data, 0, root_node, R.identity()
        )
        
        # Compare with original rotations
        for node_name, converted_rotation in frame_rotations.items():
            if motion_data.has_rotations(node_name):
                original_rotation: R = R.from_quat(motion_data.get_rotations(node_name)[frame_idx])
                
                # Calculate relative rotation
                relative_rot: R = converted_rotation * original_rotation.inv()
                angle_error = float(relative_rot.magnitude())
                
                print(f"Frame {frame_idx}, {node_name}: angular error = {angle_error:.6f} rad")
                
                # Allow for reasonable numerical error
                assert angle_error < 0.1, f"Angular error for {node_name} at frame {frame_idx} too large: {angle_error} rad"


def test_roundtrip_conversion_sample(motion_data: MotionData) -> None:
    """Test roundtrip conversion accuracy on sample frames."""
    root_node = motion_data.kinematic_tree.root
    assert root_node is not None, "Root node should exist"
    
    # Test only first few frames for performance
    test_frames: List[int] = [0, 1, 2] if motion_data.frame_count > 2 else [0]
    
    total_errors: List[float] = []
    
    for frame_idx in test_frames:
        # Step 1: Original rotational data -> positional data
        single_frame_data: MotionData = create_single_frame_motion_data(motion_data, frame_idx)
        
        if motion_data.has_positions(root_node.name):
            root_pos: np.ndarray = motion_data.get_positions(root_node.name)[frame_idx]
        else:
            root_pos = np.zeros(3)
        
        frame_positions: Dict[str, np.ndarray] = get_positions_from_rotations(
            single_frame_data, root_pos, root_node, R.identity(), scale=1
        )
        
        # Create positional motion data
        single_frame_positions: Dict[str, np.ndarray] = {}
        for node_name, position in frame_positions.items():
            if isinstance(position, np.ndarray) and position.ndim == 2:
                single_frame_positions[node_name] = position
            else:
                single_frame_positions[node_name] = position.reshape(1, -1)
        
        positional_data: MotionData = MotionData(
            motion_data.kinematic_tree,
            positions=single_frame_positions,
            rotations=None,
            frame_time=motion_data.frame_time
        )
        
        # Step 2: Positional data -> rotational data
        frame_rotations: Dict[str, R] = get_rotation_from_position(
            positional_data, 0, root_node, R.identity()
        )
        
        # Step 3: Compare original and roundtrip rotations
        for node_name, roundtrip_rotation in frame_rotations.items():
            if motion_data.has_rotations(node_name):
                original_rotation: R = R.from_quat(motion_data.get_rotations(node_name)[frame_idx])
                
                # Calculate relative rotation
                relative_rot: R = roundtrip_rotation * original_rotation.inv()
                angle_error = float(relative_rot.magnitude())
                total_errors.append(angle_error)
                
                print(f"Frame {frame_idx}, {node_name}: roundtrip error = {angle_error:.6f} rad")
    
    if total_errors:
        mean_error = float(np.mean(total_errors))
        max_error = float(np.max(total_errors))
        
        print(f"Roundtrip conversion - Mean error: {mean_error:.6f} rad, Max error: {max_error:.6f} rad")
        
        # Reasonable thresholds for roundtrip conversion
        assert mean_error < 0.1, f"Mean roundtrip error too large: {mean_error} rad"
        assert max_error < 0.5, f"Max roundtrip error too large: {max_error} rad"


def test_save_converted_bvh(motion_data: MotionData, tmp_path: Path) -> None:
    """Test saving converted motion data to BVH file."""
    root_node = motion_data.kinematic_tree.root
    assert root_node is not None, "Root node should exist"
    
    # Test with just the first few frames for performance
    test_frame_count: int = min(5, motion_data.frame_count)
    
    # Convert to positions and back to rotations (roundtrip)
    converted_positions: Dict[str, List[np.ndarray]] = {}
    converted_rotations: Dict[str, List[np.ndarray]] = {}
    
    for frame_idx in range(test_frame_count):
        # Create single-frame motion data
        single_frame_data: MotionData = create_single_frame_motion_data(motion_data, frame_idx)
        
        # Get root position
        if motion_data.has_positions(root_node.name):
            root_pos: np.ndarray = motion_data.get_positions(root_node.name)[frame_idx]
        else:
            root_pos = np.zeros(3)
        
        # Convert to positions
        frame_positions: Dict[str, np.ndarray] = get_positions_from_rotations(
            single_frame_data, root_pos, root_node, R.identity(), scale=1
        )
        
        # Store positions
        for node_name, position in frame_positions.items():
            if node_name not in converted_positions:
                converted_positions[node_name] = []
            
            if isinstance(position, np.ndarray) and position.ndim == 2:
                converted_positions[node_name].append(position[0])
            else:
                converted_positions[node_name].append(position)
        
        # Create positional motion data for this frame
        single_frame_positions: Dict[str, np.ndarray] = {}
        for node_name, position in frame_positions.items():
            if isinstance(position, np.ndarray) and position.ndim == 2:
                single_frame_positions[node_name] = position
            else:
                single_frame_positions[node_name] = position.reshape(1, -1)
        
        positional_data: MotionData = MotionData(
            motion_data.kinematic_tree,
            positions=single_frame_positions,
            rotations=None,
            frame_time=motion_data.frame_time
        )
        
        # Convert back to rotations
        frame_rotations: Dict[str, R] = get_rotation_from_position(
            positional_data, 0, root_node, R.identity()
        )
        
        # Store rotations
        for node_name, rotation in frame_rotations.items():
            if node_name not in converted_rotations:
                converted_rotations[node_name] = []
            converted_rotations[node_name].append(rotation.as_quat())
    
    # Convert to numpy arrays
    final_positions: Dict[str, np.ndarray] = {}
    for node_name in converted_positions:
        final_positions[node_name] = np.array(converted_positions[node_name])
    
    final_rotations: Dict[str, np.ndarray] = {}
    for node_name in converted_rotations:
        final_rotations[node_name] = np.array(converted_rotations[node_name])
    
    # Create final motion data
    final_motion_data: MotionData = MotionData(
        motion_data.kinematic_tree,
        positions=final_positions,
        rotations=final_rotations,
        frame_time=motion_data.frame_time
    )
    
    # Save to BVH file
    output_path: Path = tmp_path / "converted_test.bvh"
    saver: BVHSaver = BVHSaver()
    saver.save_bvh(final_motion_data, str(output_path))
    
    # Verify file was created and is not empty
    assert output_path.exists(), "Output BVH file should be created"
    assert output_path.stat().st_size > 0, "Output BVH file should not be empty"
    
    # Try to load the saved file to verify it's valid
    reloaded_data: MotionData = load_bvh(str(output_path))
    assert reloaded_data.frame_count == test_frame_count, "Frame count should match"
    assert len(reloaded_data.kinematic_tree.nodes) > 0, "Should have nodes"