import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from mocap_converter.motion_data import MotionData


def get_positions_from_rotations(
    motion_data: MotionData,
    parent_position: NDArray[np.float64],
    current_node_name: str,
    accum_rot: R = R.identity(),
    scale: float = 1,
) -> dict[str, NDArray[np.float64]]:
    if not motion_data.has_rotations(current_node_name):
        return {}

    positions: dict[str, NDArray[np.float64]] = {current_node_name: parent_position}
    rot_np = motion_data.rotations[current_node_name]
    rotation = accum_rot * R.from_quat(rot_np)

    tree = motion_data.kinematic_tree
    children = tree.get_children(current_node_name)

    if not children:
        return positions

    if len(children) == 1:  # Joint has a child
        child_node = children[0]
        offset = child_node.offset
        rotated_offset = rotation.apply(offset)
        positions[child_node.name] = parent_position + rotated_offset * scale

        r = get_positions_from_rotations(
            motion_data,
            positions[child_node.name],
            child_node.name,
            rotation,
            scale,
        )
        positions.update(r)
    else:  # Joint has children
        for child_node in children:
            offset = child_node.offset
            rotated_offset = rotation.apply(offset)
            positions[child_node.name] = parent_position + rotated_offset * scale

            r = get_positions_from_rotations(
                motion_data,
                positions[child_node.name],
                child_node.name,
                rotation,
                scale,
            )
            positions.update(r)

    return positions
