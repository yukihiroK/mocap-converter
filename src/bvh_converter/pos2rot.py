import numpy as np
from scipy.spatial.transform import Rotation as R

from bvh_converter.node import Node
from bvh_converter.motion_data import MotionData


def get_rotation_from_position(
    motion_data: MotionData,
    frame_count: int,
    current_node: Node,
    accum_rot: R = R.identity(),
) -> dict[str, R]:
    result = {}

    if current_node.children_count == 1:  # Joint has a child
        child_node = current_node.children[0]
        actual_offset = (motion_data.get_positions(child_node.name) - motion_data.get_positions(current_node.name))[
            frame_count
        ]
        initial_offset = child_node.offset
        if np.linalg.norm(initial_offset) == 0:  # child_node is an end effector
            result[current_node.name] = R.identity()
            return result

        initial_offset = initial_offset / np.linalg.norm(initial_offset)

        local_offset = accum_rot.apply(actual_offset)
        local_offset = local_offset / np.linalg.norm(local_offset)
        rotation = R.align_vectors(local_offset, initial_offset)[0]
        result[current_node.name] = rotation

        r = get_rotation_from_position(
            motion_data,
            frame_count,
            child_node,
            rotation.inv() * accum_rot,
        )
        result.update(r)

    elif current_node.children_count > 1:  # Joint has children

        actual_offsets = [
            (motion_data.get_positions(child_node.name) - motion_data.get_positions(current_node.name))[frame_count]
            for child_node in current_node.children
        ]
        initial_offsets = np.array([child_node.offset for child_node in current_node.children])
        initial_offsets = initial_offsets / np.linalg.norm(initial_offsets, axis=1)[:, np.newaxis]
        local_offsets = accum_rot.apply(np.array(actual_offsets))  # (n, 3)
        local_offsets = local_offsets / np.linalg.norm(local_offsets, axis=1)[:, np.newaxis]

        rotation = R.align_vectors(local_offsets, initial_offsets)[0]
        result[current_node.name] = rotation

        for child_node in current_node.children:
            r = get_rotation_from_position(
                motion_data,
                frame_count,
                child_node,
                rotation.inv() * accum_rot,
            )
            result.update(r)
    else:
        result[current_node.name] = R.identity()

    return result
