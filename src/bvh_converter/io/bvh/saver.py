import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import List, Dict, Optional, Literal, Tuple

from bvh_converter.node import Node
from bvh_converter.kinematic_tree import KinematicTree
from bvh_converter.motion_data import MotionData
from bvh_converter.io.bvh.node_channel import NodeChannel
from bvh_converter.io.bvh.types import CHANNEL_TYPES, NODE_TYPES, ROTATION_ORDER


class BVHSaver:
    """
    A class for saving BVH files from motion data.

    Example:

    ```python

    saver = BVHSaver()
    saver.save_bvh(motion_data, "example.bvh")

    ```

    """

    def __init__(self, rotation_orders: Dict[str, ROTATION_ORDER] = {}):
        self.rotation_orders = rotation_orders
        self.__ordered_node_channels: List[NodeChannel] = []

    def _stringify_nodes_recursive(
        self,
        node: Node,
        indent: str = "  ",
    ) -> List[str]:

        node_channel = NodeChannel.from_rotation_order(
            name=node.name,
            has_position_channels=node.is_root,
            rotation_order=self.rotation_orders.get(node.name, "ZXY"),
        )
        self.__ordered_node_channels.append(node_channel)

        if node.has_children:  # Joint/Root node with children
            child_content = []
            for child in node.children:
                child_lines = self._stringify_nodes_recursive(child, indent)
                child_content.extend(child_lines)

            node_type = "ROOT" if node.is_root else "JOINT"

            return _stringify_node(
                node_type,
                node.name,
                node.offset,
                node_channel.channels,
                child_content,
                indent,
            )

        if node.is_root:  # Root node with no children
            return _stringify_node(
                "ROOT",
                node.name,
                node.offset,
                node_channel.channels,
                indent=indent,
            )

        if node.has_siblings:  # An end-effector with siblings should be a joint node with an end site
            return _stringify_node(
                "JOINT",
                node.name,
                node.offset,
                node_channel.channels,
                children=_stringify_node("End", node.name, np.zeros(3), indent=indent),
                indent=indent,
            )

        # An end-effector with no siblings should be an end site
        return _stringify_node("End", node.name, node.offset, indent=indent)

    def _stringify_node_hierarchy(
        self,
        kinematic_tree: KinematicTree,
    ) -> str:
        if not kinematic_tree.root:
            raise ValueError("Kinematic tree does not have a node")

        lines = ["HIERARCHY"]
        lines.extend(self._stringify_nodes_recursive(kinematic_tree.root))
        return "\n".join(lines)

    def save_bvh(
        self,
        motion_data: MotionData,
        filename: str,
    ):
        hierarchy = self._stringify_node_hierarchy(motion_data.kinematic_tree)
        motion_info = _stringify_motion_info(motion_data)
        header = f"{hierarchy}\n{motion_info}"

        motion_values = _get_motion_values(self.__ordered_node_channels, motion_data)

        np.savetxt(filename, motion_values, delimiter=" ", header=header, comments="")


def _get_motion_values(
    node_channels: List[NodeChannel],
    motion_data: MotionData,
) -> np.ndarray:

    motion_values = []
    for node_channel in node_channels:
        if node_channel.has_position_channels:
            positions = motion_data.get_positions(node_channel.name)  # (frames, 3)
            motion_values.append(positions)
        if node_channel.has_rotation_channels:
            rotations = motion_data.get_rotations(node_channel.name)
            rotations = R.from_quat(rotations).as_euler(node_channel.rotation_order, degrees=True)  # (frames, 3)
            motion_values.append(rotations)

    return np.concatenate(motion_values, axis=1)  # (frames, num_channels)


def _stringify_motion_info(motion_data: MotionData) -> str:
    info_str = "MOTION\n"
    info_str += f"Frames: {motion_data.frame_count}\n"
    info_str += f"Frame Time: {motion_data.frame_time:.6f}"
    return info_str


def _stringify_node(
    node_type: NODE_TYPES,
    node_name: str,
    offset: np.ndarray,
    channels: Optional[Tuple[CHANNEL_TYPES, ...]] = None,
    children: Optional[List[str]] = None,
    indent: str = "  ",
) -> List[str]:
    lines = []
    if node_type == "End":
        lines.append("End Site")
    else:
        lines.append(f"{node_type} {node_name}")
    lines.append("{")

    lines.append(indent + f"OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")
    if channels and children:
        lines.append(indent + f"CHANNELS {len(channels)} {' '.join(channels)}")
        lines.extend([indent + line for line in children])

    lines.append("}")
    return lines
