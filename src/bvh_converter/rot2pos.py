import numpy as np
from scipy.spatial.transform import Rotation as R

from .motion_data import MotionData
from .node import Node


def get_positions_from_rotations(
    motion_data: MotionData,
    parent_position: np.ndarray,
    current_node: Node,
    accum_rot: R = R.identity(),
    scale: float = 1,
) -> dict[str, np.ndarray]:
    if not motion_data.has_rotations(current_node.name):
        return {}

    positions: dict[str, np.ndarray] = {current_node.name: parent_position}
    rot_np = motion_data.get_rotations(current_node.name)
    rotation = accum_rot * R.from_quat(rot_np)

    if not current_node.has_children:
        return positions

    if current_node.children_count == 1:  # Joint has a child
        child_node = current_node.children[0]
        offset = child_node.offset
        rotated_offset = rotation.apply(offset)
        positions[child_node.name] = parent_position + rotated_offset * scale

        r = get_positions_from_rotations(
            motion_data,
            positions[child_node.name],
            child_node,
            rotation,
            scale,
        )
        positions.update(r)
    else:  # Joint has children
        for child_node in current_node.children:
            offset = child_node.offset
            rotated_offset = rotation.apply(offset)
            positions[child_node.name] = parent_position + rotated_offset * scale

            r = get_positions_from_rotations(
                motion_data,
                positions[child_node.name],
                child_node,
                rotation,
                scale,
            )
            positions.update(r)

    return positions
