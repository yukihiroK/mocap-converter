from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from bvh_converter.io.bvh.node_channel import NodeChannel
from bvh_converter.io.bvh.types import CHANNEL_TYPES, NODE_TYPES, ROTATION_ORDER
from bvh_converter.kinematic_tree import KinematicTree
from bvh_converter.motion_data import MotionData


def save_bvh(
    motion_data: MotionData,
    filename: str,
    rotation_orders: dict[str, ROTATION_ORDER] | None = None,
) -> None:
    """
    Save BVH file from motion data.
    
    Args:
        motion_data: MotionData containing kinematic tree and motion information
        filename: Path to save the BVH file
        rotation_orders: Optional mapping of node names to rotation orders
        
    Example:
        >>> save_bvh(motion_data, "example.bvh")
        >>> save_bvh(motion_data, "example.bvh", {"root": "ZXY"})
    """
    if rotation_orders is None:
        rotation_orders = {}
    
    hierarchy, ordered_node_channels = _build_hierarchy_string(motion_data.kinematic_tree, rotation_orders)
    motion_info = _build_motion_info_string(motion_data)
    header = f"{hierarchy}\n{motion_info}"

    motion_values = _extract_motion_values(ordered_node_channels, motion_data)

    np.savetxt(filename, motion_values, delimiter=" ", header=header, comments="")

def _build_nodes_recursive(
    tree: KinematicTree,
    node_name: str,
    rotation_orders: dict[str, ROTATION_ORDER],
    ordered_channels: list[NodeChannel],
    indent: str = "  ",
) -> list[str]:
    """Recursively build BVH node strings and track channel order."""
    node = tree.get_node(node_name)
    node_channel = NodeChannel.from_rotation_order(
        name=node.name,
        has_position_channels=node.is_root,
        rotation_order=rotation_orders.get(node.name, "ZXY"),
    )
    ordered_channels.append(node_channel)

    children = tree.get_children(node_name)
    if children:  # Joint/Root node with children
        child_content: list[str] = []
        for child in children:
            child_lines = _build_nodes_recursive(tree, child.name, rotation_orders, ordered_channels, indent)
            child_content.extend(child_lines)

        node_type = "ROOT" if node.is_root else "JOINT"

        return _build_node_string(
            node_type,
            node.name,
            node.offset,
            node_channel.channels,
            child_content,
            indent,
        )

    if node.is_root:  # Root node with no children
        return _build_node_string(
            "ROOT",
            node.name,
            node.offset,
            node_channel.channels,
            indent=indent,
        )

    if tree.has_siblings(node_name):  # An end-effector with siblings should be a joint node with an end site
        return _build_node_string(
            "JOINT",
            node.name,
            node.offset,
            node_channel.channels,
            children=_build_node_string("End", node.name, np.zeros(3), indent=indent),
            indent=indent,
        )

    # An end-effector with no siblings should be an end site
    return _build_node_string("End", node.name, node.offset, indent=indent)


def _build_hierarchy_string(
    kinematic_tree: KinematicTree,
    rotation_orders: dict[str, ROTATION_ORDER],
) -> tuple[str, list[NodeChannel]]:
    """Build hierarchy string and return ordered node channels."""
    root = kinematic_tree.root
    if root is None:
        raise ValueError("Kinematic tree does not have a root node")

    ordered_node_channels: list[NodeChannel] = []
    lines = ["HIERARCHY"]
    lines.extend(_build_nodes_recursive(kinematic_tree, root.name, rotation_orders, ordered_node_channels))
    return "\n".join(lines), ordered_node_channels


def _extract_motion_values(
    node_channels: list[NodeChannel],
    motion_data: MotionData,
) -> NDArray[np.float64]:
    """Extract motion values from motion data based on node channels."""
    kinematic_tree = motion_data.kinematic_tree
    motion_values: list[NDArray[np.float64]] = []
    
    for node_channel in node_channels:
        # Check if this is a leaf node with no siblings through the tree
        is_leaf = kinematic_tree.is_leaf(node_channel.name)
        has_siblings = kinematic_tree.has_siblings(node_channel.name)
        if is_leaf and not has_siblings:
            continue

        if node_channel.has_position_channels and motion_data.has_positions(node_channel.name):
            positions = motion_data.get_positions(node_channel.name)  # (frames, 3)
            motion_values.append(positions)
        if node_channel.has_rotation_channels and motion_data.has_rotations(node_channel.name):
            rotations = motion_data.get_rotations(node_channel.name)
            euler_rotations = R.from_quat(rotations).as_euler(node_channel.rotation_order, degrees=True)  # (frames, 3)
            motion_values.append(euler_rotations)

    if not motion_values:
        raise ValueError("No motion data found to save")
    
    return np.concatenate(motion_values, axis=1)  # (frames, num_channels)


def _build_motion_info_string(motion_data: MotionData) -> str:
    """Build motion information string."""
    return (
        f"MOTION\n"
        f"Frames: {motion_data.frame_count}\n"
        f"Frame Time: {motion_data.frame_time:.6f}"
    )


def _build_node_string(
    node_type: NODE_TYPES,
    node_name: str,
    offset: NDArray[np.float64],
    channels: tuple[CHANNEL_TYPES, ...] | None = None,
    children: list[str] | None = None,
    indent: str = "  ",
) -> list[str]:
    """Build string representation of a single BVH node."""
    lines: list[str] = []
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
