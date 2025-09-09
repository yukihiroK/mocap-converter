from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from mocap_converter.io.bvh.channel_layout import BVHChannelLayout
from mocap_converter.io.bvh.types import CHANNEL_TYPES, NODE_TYPES
from mocap_converter.kinematic_tree import KinematicTree
from mocap_converter.motion_data import MotionData


def save_bvh(
    motion_data: MotionData,
    filename: str,
    channel_layouts: dict[str, BVHChannelLayout] | None = None,
) -> None:
    """
    Save BVH file from motion data.

    Args:
        motion_data: MotionData containing kinematic tree and motion information
        filename: Path to save the BVH file
        channel_layouts: Optional explicit per-node channels. When absent for a node,
            a default layout is used: root has XYZ position channels; all non-End nodes have
            three rotation channels with order "ZXY".

    Example:
        >>> save_bvh(motion_data, "example.bvh")
        >>> # Override per-node layout (e.g., root with XYZ positions + ZYX rotations)
        >>> layout = BVHChannelLayout.from_rotation_order("ZYX", has_position_channels=True)
        >>> save_bvh(motion_data, "example.bvh", {"Hips": layout})
    """
    if channel_layouts is None:
        channel_layouts = {}

    hierarchy, node_order = _build_hierarchy_string(motion_data.kinematic_tree, channel_layouts)
    motion_info = _build_motion_info_string(motion_data)
    header = f"{hierarchy}\n{motion_info}"

    motion_values = _extract_motion_values(node_order, motion_data, channel_layouts)

    np.savetxt(filename, motion_values, delimiter=" ", header=header, comments="")


def _get_channel_layout_for_node(
    tree: KinematicTree,
    node_name: str,
    channel_layouts: dict[str, BVHChannelLayout],
) -> BVHChannelLayout:
    node = tree.get_node(node_name)
    explicit = channel_layouts.get(node.name)
    if explicit is not None:
        return explicit

    return BVHChannelLayout.from_rotation_order(
        has_position_channels=node.is_root,
        rotation_order="ZXY",
    )


def _build_nodes_recursive(
    tree: KinematicTree,
    node_name: str,
    channel_layouts: dict[str, BVHChannelLayout],
    node_order: list[str],
    indent: str = "  ",
) -> list[str]:
    """Recursively build BVH node strings and track channel order."""
    node = tree.get_node(node_name)
    channel_layout = _get_channel_layout_for_node(tree, node_name, channel_layouts)

    children = tree.get_children(node_name)
    if children:  # Joint/Root node with children
        child_content: list[str] = []
        node_order.append(node.name)
        for child in children:
            child_lines = _build_nodes_recursive(tree, child.name, channel_layouts, node_order, indent)
            child_content.extend(child_lines)
        node_type = "ROOT" if node.is_root else "JOINT"

        return _build_node_string(
            node_type,
            node.name,
            node.offset,
            channel_layout.channels,
            child_content,
            indent,
        )

    if node.is_root:  # Root node with no children
        node_order.append(node.name)
        return _build_node_string(
            "ROOT",
            node.name,
            node.offset,
            channel_layout.channels,
            indent=indent,
        )

    # Non-root node with no children
    if tree.has_siblings(node_name):
        node_order.append(node.name)
        return _build_node_string(
            "JOINT",
            node.name,
            node.offset,
            channel_layout.channels,
            children=_build_node_string("End", node.name, np.zeros(3), indent=indent),
            indent=indent,
        )

    # An end-effector with no siblings should be an End Site
    return _build_node_string("End", node.name, node.offset, indent=indent)


def _build_hierarchy_string(
    kinematic_tree: KinematicTree,
    channel_layouts: dict[str, BVHChannelLayout],
) -> tuple[str, list[str]]:
    """Build hierarchy string and return ordered node names for motion columns."""
    root = kinematic_tree.root
    if root is None:
        raise ValueError("Kinematic tree does not have a root node")

    ordered_node_names: list[str] = []
    lines = ["HIERARCHY"]
    lines.extend(_build_nodes_recursive(kinematic_tree, root.name, channel_layouts, ordered_node_names))
    return "\n".join(lines), ordered_node_names


def _extract_motion_values(
    node_order: list[str], motion_data: MotionData, channel_layouts: dict[str, BVHChannelLayout]
) -> NDArray[np.float64]:
    """Extract motion values from motion data based on node channels."""
    kinematic_tree = motion_data.kinematic_tree
    motion_values: list[NDArray[np.float64]] = []

    for node_name in node_order:
        channel_layout = _get_channel_layout_for_node(kinematic_tree, node_name, channel_layouts)

        # Enforce consistency between declared channels and provided data
        if channel_layout.has_position_channels and not motion_data.has_positions(node_name):
            raise ValueError(f"Position channels declared for '{node_name}' but no position data present in MotionData")
        if channel_layout.has_rotation_channels and not motion_data.has_rotations(node_name):
            raise ValueError(f"Rotation channels declared for '{node_name}' but no rotation data present in MotionData")

        if channel_layout.has_position_channels:
            positions = motion_data.positions[node_name]  # (frames, 3)
            motion_values.append(positions)
        if channel_layout.has_rotation_channels:
            rotations = motion_data.rotations[node_name]
            euler_rotations = R.from_quat(rotations).as_euler(
                channel_layout.rotation_order, degrees=True
            )  # (frames, 3)
            motion_values.append(euler_rotations)

    if not motion_values:
        raise ValueError("No motion data found to save")

    return np.concatenate(motion_values, axis=1)  # (frames, num_channels)


def _build_motion_info_string(motion_data: MotionData) -> str:
    """Build motion information string."""
    return f"MOTION\n" f"Frames: {motion_data.frame_count}\n" f"Frame Time: {motion_data.frame_time:.6f}"


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
    if channels:
        lines.append(indent + f"CHANNELS {len(channels)} {' '.join(channels)}")
    if children:
        lines.extend([indent + line for line in children])

    lines.append("}")
    return lines
