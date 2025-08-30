from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from bvh_converter.io.bvh.channel_layout import BVHChannelLayout
from bvh_converter.io.bvh.types import CHANNEL_TYPES, validate_channel
from bvh_converter.kinematic_tree import KinematicTree
from bvh_converter.motion_data import MotionData
from bvh_converter.node import Node


# === Immutable Data Structures ===
@dataclass(frozen=True)
class BVHSections:
    hierarchy_lines: tuple[str, ...]
    motion_lines: tuple[str, ...]


@dataclass(frozen=True)
class ParsedHierarchy:
    kinematic_tree: KinematicTree
    node_channels: MappingProxyType[str, BVHChannelLayout]


@dataclass(frozen=True)
class ParsedMotion:
    hierarchy: ParsedHierarchy
    frame_count: int
    frame_time: float
    frames: tuple[tuple[float, ...], ...]


class ParseError(Exception):
    """Raised when an error occurs while parsing a BVH file"""


# === Core Parsing Functions ===


def split_into_sections(content: str) -> BVHSections:
    """Split BVH content into hierarchy and motion sections"""
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    motion_index = None
    for i, line in enumerate(lines):
        if line == "MOTION":
            motion_index = i
            break

    if motion_index is None:
        raise ParseError("MOTION section not found in BVH file")

    hierarchy_lines = tuple(lines[:motion_index])
    motion_lines = tuple(lines[motion_index:])

    return BVHSections(hierarchy_lines, motion_lines)


def parse_hierarchy_section(sections: BVHSections) -> ParsedHierarchy:
    """Parse the hierarchy section to extract nodes and channels"""
    nodes: list[Node] = []
    node_channels: dict[str, BVHChannelLayout] = {}
    node_stack: list[str] = []
    current_node_name: str | None = None

    for line_num, line in enumerate(sections.hierarchy_lines):
        tokens = line.split()
        if not tokens:
            continue

        if tokens[0] in ("ROOT", "JOINT", "End"):
            node = _parse_node_line(current_node_name, tokens)
            nodes.append(node)
            node_stack.append(node.name)
            current_node_name = node.name

        elif tokens[0] == "OFFSET":
            if current_node_name is None:
                raise ParseError("OFFSET specified before node definition", line_num)

            offset = _parse_offset_line(tokens[1:])
            # Update the node with offset
            nodes[-1] = nodes[-1].copy_with(offset=offset)

        elif tokens[0] == "}":
            if not node_stack:
                raise ParseError("'}' found without corresponding node", line_num)
            node_stack.pop()
            current_node_name = node_stack[-1] if node_stack else None

        elif tokens[0] == "CHANNELS":
            if current_node_name is None:
                raise ParseError("CHANNELS specified before node definition", line_num)

            channel_layout = _parse_channels_line(tokens)
            node_channels[current_node_name] = channel_layout

    return ParsedHierarchy(KinematicTree.from_nodes(nodes), MappingProxyType(node_channels))


def parse_motion_data(hierarchy: ParsedHierarchy, sections: BVHSections) -> ParsedMotion:
    motion_lines = sections.motion_lines[1:]  # Skip "MOTION" line

    if len(motion_lines) < 2:
        raise ParseError("Invalid motion section: missing frame data")

    # Parse frame count
    frames_line = motion_lines[0].split()
    if len(frames_line) < 2 or frames_line[0] != "Frames:":
        raise ParseError("Invalid frame count specification")

    try:
        frame_count = int(frames_line[1])
    except ValueError:
        raise ParseError(f"Invalid frame count: {frames_line[1]}")

    # Parse frame time
    time_line = motion_lines[1].split()
    if len(time_line) < 3 or time_line[0] != "Frame" or time_line[1] != "Time:":
        raise ParseError("Invalid frame time specification")

    try:
        frame_time = float(time_line[2])
    except ValueError:
        raise ParseError(f"Invalid frame time: {time_line[2]}")

    # Parse frame data
    frame_data_lines = motion_lines[2:]
    frames: list[tuple[float, ...]] = []

    expected_channels = sum(ch.channel_count for ch in hierarchy.node_channels.values())

    for frame_line in frame_data_lines:
        if not frame_line.strip():
            continue

        try:
            values = tuple(map(float, frame_line.split()))
            if len(values) != expected_channels:
                raise ParseError(f"Expected {expected_channels} values, got {len(values)}")
            frames.append(values)
        except ValueError as e:
            raise ParseError(f"Invalid frame data: {str(e)}")

    if len(frames) != frame_count:
        raise ParseError(f"Frame count mismatch: expected {frame_count}, got {len(frames)}")

    return ParsedMotion(hierarchy, frame_count, frame_time, tuple(frames))


def build_motion_data_efficiently(parsed: ParsedMotion) -> MotionData:
    """Build MotionData with O(N) efficiency using pre-allocated arrays"""

    # Pre-allocate arrays based on frame count
    position_arrays: dict[str, NDArray[np.float64]] = {}
    rotation_arrays: dict[str, NDArray[np.float64]] = {}

    for node_name, channel_layout in parsed.hierarchy.node_channels.items():
        if channel_layout.has_position_channels:
            position_arrays[node_name] = np.empty((parsed.frame_count, 3), dtype=np.float64)
        if channel_layout.has_rotation_channels:
            rotation_arrays[node_name] = np.empty((parsed.frame_count, 4), dtype=np.float64)

    # Populate arrays directly (O(N) instead of O(N²))
    for frame_idx, frame_data in enumerate(parsed.frames):
        offset = 0
        for node_name, channel_layout in parsed.hierarchy.node_channels.items():
            values = frame_data[offset : offset + channel_layout.channel_count]
            position, rotation = _parse_channel_values(channel_layout.channels, values)

            if channel_layout.has_position_channels:
                position_arrays[node_name][frame_idx] = position
            if channel_layout.has_rotation_channels:
                rotation_arrays[node_name][frame_idx] = rotation.as_quat()

            offset += channel_layout.channel_count

    return MotionData(parsed.hierarchy.kinematic_tree, position_arrays, rotation_arrays, parsed.frame_time)


# === Helper Functions ===


def _parse_node_line(current_node_name: str | None, tokens: list[str]) -> Node:
    """Parse a node declaration line"""
    node_type, node_name = tokens[0], tokens[1]

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
    return Node(name=node_name, parent_name=parent_name)


def _parse_offset_line(offset_tokens: list[str]) -> NDArray[np.float64]:
    """Parse OFFSET line"""
    try:
        return np.array([float(x) for x in offset_tokens], dtype=np.float64)
    except ValueError:
        raise ParseError(f"Invalid OFFSET values: {' '.join(offset_tokens)}")


def _parse_channels_line(tokens: list[str]) -> BVHChannelLayout:
    """Parse CHANNELS line"""
    try:
        channel_count = int(tokens[1])
        channel_names = tokens[2:]

        if len(channel_names) != channel_count:
            raise ParseError(f"Channel count mismatch: expected {channel_count}, got {len(channel_names)}")

        validated_channels: list[CHANNEL_TYPES] = []
        for ch in channel_names:
            validated_channels.append(validate_channel(ch))
        return BVHChannelLayout.from_bvh_channels(tuple(validated_channels))
    except (ValueError, IndexError) as e:
        raise ParseError(f"Invalid CHANNELS specification: {str(e)}")


def load_bvh(filename: str) -> MotionData:
    """
    Load BVH file and extract motion data from it using functional programming approach.

    Args:
        filename: Path to the BVH file to load

    Returns:
        MotionData containing kinematic tree and motion information

    Raises:
        ParseError: When an error occurs while parsing the BVH file
    """
    content = Path(filename).read_text()
    return parse_bvh_content(content)


def parse_bvh_content(content: str) -> MotionData:
    """Pure functional BVH parsing pipeline"""
    sections_result = split_into_sections(content)
    parsed_hierarchy = parse_hierarchy_section(sections_result)
    parsed_motion = parse_motion_data(parsed_hierarchy, sections_result)
    motion = build_motion_data_efficiently(parsed_motion)
    return motion


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
