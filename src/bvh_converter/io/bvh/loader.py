import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from bvh_converter.io.bvh.node_channel import NodeChannel
from bvh_converter.io.bvh.types import CHANNEL_TYPES, NODE_TYPES, validate_channel
from bvh_converter.kinematic_tree import KinematicTree
from bvh_converter.motion_data import MotionData
from bvh_converter.node import Node


class ParseError(Exception):
    """Raised when an error occurs while parsing a BVH file"""


def load_bvh(filename: str) -> MotionData:
    """
    Load BVH file and extract motion data from it.

    Args:
        filename: Path to the BVH file to load

    Returns:
        MotionData containing kinematic tree and motion information

    Raises:
        ParseError: When an error occurs while parsing the BVH file
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    nodes: dict[str, Node] = {}
    node_stack: list[str] = []
    current_node_name: str | None = None
    node_channels: list[NodeChannel] = []
    channel_count = 0
    position_values: dict[str, list[NDArray[np.float64]]] = {}
    rotation_values: dict[str, list[NDArray[np.float64]]] = {}
    frame_time = 0.0

    for i, line in enumerate(lines):
        tokens = line.strip().split()

        if not tokens:  # Skip empty lines
            continue

        if tokens[0] in ("ROOT", "JOINT", "End"):
            node = _parse_node(current_node_name, tokens[0], tokens[1])
            nodes[node.name] = node
            node_stack.append(node.name)
            current_node_name = node.name

        elif tokens[0] == "OFFSET":
            if current_node_name is None:
                raise ParseError("OFFSET specified before ROOT or JOINT is defined")

            node_offset = _parse_node_offset(current_node_name, tokens[1:])
            current_node = nodes[current_node_name]
            nodes[current_node_name] = current_node.copy_with(offset=node_offset)

        elif tokens[0] == "}":
            if not node_stack:
                raise ParseError("'}' found without a corresponding node")

            node_stack.pop()
            current_node_name = node_stack[-1] if node_stack else None

        elif tokens[0] == "CHANNELS":
            node_channel = _parse_node_channels(current_node_name, tokens[2:])

            node_channels.append(node_channel)
            channel_count += node_channel.channel_count

            if node_channel.has_position_channels:
                position_values[node_channel.name] = []
            if node_channel.has_rotation_channels:
                rotation_values[node_channel.name] = []

        elif tokens[0] == "MOTION":
            if not nodes:
                raise ParseError("No nodes defined before MOTION data")

            (frame_time, position_values, rotation_values) = _parse_motion_data(
                channel_count, node_channels, position_values, rotation_values, lines[i + 1 :]
            )
            break

    kinematic_tree = KinematicTree.from_nodes(list(nodes.values()))
    position_data: dict[str, NDArray[np.float64]] = {
        name: np.array(positions) for name, positions in position_values.items()
    }
    rotation_data: dict[str, NDArray[np.float64]] = {
        name: np.array(rotations) for name, rotations in rotation_values.items()
    }

    return MotionData(kinematic_tree, position_data, rotation_data, frame_time)


def _parse_node(current_node_name: str | None, node_type: NODE_TYPES, node_name: str) -> Node:
    """Parse a node declaration and update state."""
    if node_type == "ROOT" and current_node_name is not None:
        raise ParseError("Multiple ROOT nodes are not allowed")

    elif node_type == "JOINT" and current_node_name is None:
        raise ParseError("JOINT specified before ROOT is defined")

    elif node_type == "End":
        if current_node_name is None:
            raise ParseError("End node specified before ROOT or JOINT is defined")

        if node_name != "Site":
            raise ParseError(f'"End {node_name}" is not a valid node type (expected "End Site")')

        node_name = f"{current_node_name}_EndSite"

    parent_name = current_node_name if node_type != "ROOT" else None
    node = Node(name=node_name, parent_name=parent_name)
    return node


def _parse_node_offset(current_node_name: str | None, offset_tokens: list[str]) -> NDArray[np.float64]:
    """Parse node offset and update the current node."""

    try:
        offset_values = tuple(map(float, offset_tokens))
    except ValueError:
        raise ParseError(f"Invalid OFFSET values: {' '.join(offset_tokens)}")

    return np.array(offset_values, dtype=np.float64)


def _parse_node_channels(current_node_name: str | None, channel_tokens: list[str]) -> NodeChannel:
    """Parse node channels and update state."""
    if current_node_name is None:
        raise ParseError("CHANNELS specified before ROOT or JOINT is defined")

    try:
        valid_channels_list: list[CHANNEL_TYPES] = []
        for channel in channel_tokens:
            validated_channel = validate_channel(channel)
            valid_channels_list.append(validated_channel)
        valid_channels = tuple(valid_channels_list)
    except ValueError as e:
        raise ParseError(str(e))

    node_channel = NodeChannel.from_channels(current_node_name, valid_channels)
    return node_channel


def _parse_motion_data(
    channel_count: int,
    node_channels: list[NodeChannel],
    position_data: dict[str, list[NDArray[np.float64]]],
    rotation_data: dict[str, list[NDArray[np.float64]]],
    lines: list[str],
) -> tuple[float, dict[str, list[NDArray[np.float64]]], dict[str, list[NDArray[np.float64]]]]:
    """Parse motion data section."""

    frame_time = _parse_frame_time(lines[1])

    for line in lines[2:]:
        frame_data = line.strip().split()
        if not frame_data:
            continue
        (position_data, rotation_data) = _parse_frame_data(
            channel_count, node_channels, position_data, rotation_data, frame_data
        )

    return (frame_time, position_data, rotation_data)


def _parse_frame_data(
    channel_count: int,
    node_channels: list[NodeChannel],
    position_data: dict[str, list[NDArray[np.float64]]],
    rotation_data: dict[str, list[NDArray[np.float64]]],
    frame_data: list[str],
) -> tuple[dict[str, list[NDArray[np.float64]]], dict[str, list[NDArray[np.float64]]]]:
    """Parse a single frame of motion data."""
    try:
        channel_values = tuple(map(float, frame_data))
    except ValueError:
        raise ParseError(f"Invalid values in frame data: {' '.join(frame_data)}")

    if len(channel_values) != channel_count:
        raise ParseError(f"Expected {channel_count} channel values, got {len(channel_values)} instead")

    new_position_data = {name: data.copy() for name, data in position_data.items()}
    new_rotation_data = {name: data.copy() for name, data in rotation_data.items()}

    offset = 0
    for node_channel in node_channels:
        values = channel_values[offset : offset + node_channel.channel_count]
        position, rotation = _parse_channel_values(node_channel.channels, values)

        if node_channel.has_position_channels:
            new_position_data[node_channel.name].append(position)
        if node_channel.has_rotation_channels:
            new_rotation_data[node_channel.name].append(rotation.as_quat())  # scalar-last

        offset += node_channel.channel_count

    return (new_position_data, new_rotation_data)


def _parse_frame_time(line: str) -> float:
    """Parse frame time from MOTION section."""
    tokens = line.strip().split()
    if len(tokens) < 3 or tokens[0] != "Frame" or tokens[1] != "Time:":
        raise ParseError("Frame time is not specified")

    try:
        frame_time = float(tokens[2])
    except ValueError:
        raise ParseError(f"Invalid frame time: {tokens[2]}")

    return frame_time


def _parse_channel_values(
    channels: tuple[CHANNEL_TYPES, ...], values: tuple[float, ...]
) -> tuple[NDArray[np.float64], R]:
    """Parse channel values into position and rotation data."""
    if len(channels) != len(values):
        raise ParseError("Number of channels and values do not match")

    position = np.zeros(3, dtype=np.float64)
    rot_order = ""
    rot_values: list[float] = []

    for channel, value in zip(channels, values):
        if channel == "Xposition":
            position[0] = value
        elif channel == "Yposition":
            position[1] = value
        elif channel == "Zposition":
            position[2] = value
        elif channel == "Xrotation":
            rot_order += "X"
            rot_values.append(value)
        elif channel == "Yrotation":
            rot_order += "Y"
            rot_values.append(value)
        elif channel == "Zrotation":
            rot_order += "Z"
            rot_values.append(value)

    rotation = R.from_euler(rot_order, rot_values, degrees=True) if rot_order else R.identity()
    return position, rotation
