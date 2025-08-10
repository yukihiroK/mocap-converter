import numpy as np
from scipy.spatial.transform import Rotation as R

from bvh_converter.motion_data import MotionData
from bvh_converter.node import Node


def get_align_rotations(
    local_offsets: np.ndarray,
    initial_offset: np.ndarray,
) -> R:
    """
    Align local offsets to initial offsets using R.align_vectors.
    Args:
        local_offsets: normalized local offsets (shape: (n_frames, n_samples, 3)).
        initial_offset: normalized initial offset (shape: (n_samples, 3)).
    Returns:
        Rotations (shape: (n_frames, 4)) in quaternion format.
    """
    n_frames = local_offsets.shape[0]
    rotations: list[R] = []
    from_vecs = initial_offset.reshape(-1, 3)  # Reshape to (n_samples, 3)

    for i in range(n_frames):
        to_vecs = local_offsets[i].reshape(-1, 3)  # Reshape to (n_samples, 3)
        align_result = R.align_vectors(to_vecs, from_vecs)
        rotation = align_result[0]  # (1, 4)
        rotations.append(rotation)

    return R.concatenate(rotations)


def apply_rotations(
    rot: np.ndarray,  # (n,4)
    vec: np.ndarray,  # (n,m,3)
) -> np.ndarray:  # (n,m,3)
    """Apply rotations to vectors using quaternion representation.
    Args:
        rot_xyzw: Array of quaternions (shape: (n, 4)).
        vec: Array of vectors (shape: (n, m, 3)).
    Returns:
        Rotated vectors (shape: (n, m, 3)).
    """

    n, m, _ = vec.shape

    vec = vec.reshape(n * m, 3)
    rot = np.repeat(rot, m, axis=0)  # (n*m,4)
    ans = R.from_quat(rot).apply(vec)  # (n*m,3)
    return ans.reshape(n, m, 3)


def get_rotations_from_positions(
    motion_data: MotionData,
    current_node: Node,
    accum_rot: R | None = None,  # (n_frames, 3, 3) or (3, 3)
) -> dict[str, R]:
    if accum_rot is None:
        accum_rot = R.identity(motion_data.frame_count)

    result = {}

    if current_node.children_count == 1:  # Joint has a child
        child_node = current_node.children[0]
        actual_offsets = motion_data.get_positions(child_node.name) - motion_data.get_positions(current_node.name)

        initial_offset = child_node.offset  # (3,)
        if np.linalg.norm(initial_offset) == 0:  # child_node is an end effector
            result[current_node.name] = R.identity(motion_data.frame_count)
            return result

        initial_offset = initial_offset / np.linalg.norm(initial_offset)

        local_offsets = accum_rot.apply(actual_offsets)  # (num_frames, 3) -> (num_frames, 3)
        local_offsets = local_offsets / np.linalg.norm(local_offsets, axis=1)[:, np.newaxis]  # Normalize

        rotations = get_align_rotations(local_offsets.reshape(-1, 1, 3), initial_offset)
        result[current_node.name] = rotations

        r = get_rotations_from_positions(
            motion_data,
            child_node,
            rotations.inv() * accum_rot,
        )
        result.update(r)

    elif current_node.children_count > 1:  # Joint has children
        actual_offsets = [
            (motion_data.get_positions(child_node.name) - motion_data.get_positions(current_node.name))  # [frame_count]
            for child_node in current_node.children
        ]  # (n_children, n_frames, 3)
        initial_offsets = np.array([child_node.offset for child_node in current_node.children])  # (n_children, 3)
        initial_offsets = initial_offsets / np.linalg.norm(initial_offsets, axis=1)[:, np.newaxis]
        local_offsets = apply_rotations(
            accum_rot.as_quat(),  # (n_frames, 4)
            np.array(actual_offsets).transpose(1, 0, 2),  # (n_frames, n_children, 3)
        )  # (n_frames, n_children, 3)

        # Get rotations for all frames
        rotations = get_align_rotations(
            local_offsets, initial_offsets
        )  # (n_frames, n_children, 3), (n_children, 3) -> (n_frames, 4)
        result[current_node.name] = rotations

        for child_node in current_node.children:
            r = get_rotations_from_positions(
                motion_data,
                child_node,
                rotations.inv() * accum_rot,
            )
            result.update(r)
    else:
        result[current_node.name] = R.identity(motion_data.frame_count)

    return result
