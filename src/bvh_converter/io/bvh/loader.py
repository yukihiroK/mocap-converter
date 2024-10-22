import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import List, Dict, Optional, Literal, Tuple

from bvh_converter.node import Node
from bvh_converter.kinematic_tree import KinematicTree
from bvh_converter.motion_data import MotionData
from bvh_converter.io.bvh.types import CHANNEL_TYPES, NODE_TYPES, validate_channel, NodeChannel


def load_bvh(filename: str) -> MotionData:
    loader = BVHLoader()
    return loader.load_bvh(filename)


class BVHLoader:
    """
    A class for loading BVH files and extracting motion data from them.
    
    Example usage:

    loader = BVHLoader() \\
    motion_data = loader.load_bvh("example.bvh")
       
    """

    class ParseError(Exception):
        """Raised when an error occurs while parsing a BVH file"""

        pass

    def __init_state(self):
        self.__nodes: Dict[str, Node] = {}
        self.__current_node: Optional[Node] = None
        self.__node_stack: List[Node] = []

        self.__channel_count = 0
        self.__node_channels: List[NodeChannel] = []
        self.__frame_time = 0.0
        self.__position_data: Dict[str, List[np.ndarray]] = {}
        self.__rotation_data: Dict[str, List[np.ndarray]] = {}

    def load_bvh(self, filename: str) -> MotionData:
        self.__init_state()

        with open(filename, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            tokens = line.strip().split()

            if not tokens:  # Skip empty lines
                continue

            if tokens[0] == "ROOT" or tokens[0] == "JOINT" or tokens[0] == "End":
                self.__parse_node(tokens[0], tokens[1])

            elif tokens[0] == "OFFSET":
                self.__parse_node_offset(tokens[1:])

            elif tokens[0] == "}":
                self.__pop_node()

            elif tokens[0] == "CHANNELS":
                self.__parse_node_channels(tokens[2:])

            elif tokens[0] == "MOTION":
                self.__parse_motion_data(lines[i + 1 :])
                break

        kinematic_tree = KinematicTree(list(self.__nodes.values()))
        position_data = {name: np.array(positions) for name, positions in self.__position_data.items()}
        rotation_data = {name: np.array(rotations) for name, rotations in self.__rotation_data.items()}

        return MotionData(kinematic_tree, position_data, rotation_data, self.__frame_time)

    def __push_node(self, node: Node):
        if node.name in self.__nodes:
            raise self.ParseError(f"Node with name {node.name} already exists")

        self.__nodes[node.name] = node
        self.__node_stack.append(node)

        if self.__current_node:
            self.__current_node._add_child(node)
        self.__current_node = node

    def __pop_node(self):
        if not self.__node_stack:
            raise self.ParseError("'}' found without a corresponding node")

        self.__node_stack.pop()
        if self.__node_stack:
            self.__current_node = self.__node_stack[-1]
        else:
            self.__current_node = None

    def __parse_node(self, node_type: NODE_TYPES, node_name: str):
        if node_type == "ROOT" and self.__nodes:
            raise self.ParseError("Multiple ROOT nodes are not allowed")

        elif node_type == "JOINT" and self.__current_node is None:
            raise self.ParseError("JOINT specified before ROOT is defined")

        elif node_type == "End":
            if self.__current_node is None:
                raise self.ParseError("End node specified before ROOT or JOINT is defined")

            if not node_name == "Site":
                raise self.ParseError(f'"End {node_name}" is not a valid node type (expected "End Site")')

            node_name = self.__get_end_site_name(self.__current_node.name)

        self.__push_node(Node(node_name, parent=self.__current_node))

    def __parse_node_offset(self, offset: List[str]):
        if self.__current_node is None:
            raise self.ParseError("OFFSET specified before ROOT or JOINT is defined")

        try:
            offset_values = list(map(float, offset))
        except ValueError:
            raise self.ParseError(f"Invalid OFFSET values: {' '.join(offset)}")

        self.__current_node.offset = np.array(offset_values)

    def __parse_node_channels(self, channels: List[str]):
        if self.__current_node is None:
            raise self.ParseError("CHANNELS specified before ROOT or JOINT is defined")

        try:
            valid_channels = list(map(validate_channel, channels))
        except ValueError as e:
            raise self.ParseError(str(e))

        current_node_name = self.__current_node.name
        self.__node_channels.append(NodeChannel(current_node_name, valid_channels))
        self.__channel_count += len(valid_channels)

        if any("position" in channel for channel in valid_channels):
            self.__position_data[current_node_name] = []
        if any("rotation" in channel for channel in valid_channels):
            self.__rotation_data[current_node_name] = []

    def __parse_motion_data(self, lines: List[str]):
        if not self.__nodes:
            raise self.ParseError("No nodes defined before MOTION data")

        num_frames = self.__parse_frame_num(lines[0])
        self.__frame_time = self.__parse_frame_time(lines[1])

        for line in lines[2:]:
            frame_data = line.strip().split()
            if not frame_data:
                continue

            self.__parse_frame_data(frame_data)

    def __parse_frame_num(self, line: str):
        tokens = line.strip().split()
        if tokens[0] != "Frames:":
            raise self.ParseError("The number of frames is not specified")

        try:
            num_frames = int(tokens[1])
        except ValueError:
            raise self.ParseError(f"Invalid number of frames: {tokens[1]}")

        return num_frames

    def __parse_frame_time(self, line: str):
        tokens = line.strip().split()
        if tokens[0] != "Frame" or tokens[1] != "Time:":
            raise self.ParseError("Frame time is not specified")

        try:
            frame_time = float(tokens[2])
        except ValueError:
            raise self.ParseError(f"Invalid frame time: {tokens[2]}")

        return frame_time

    def __parse_frame_data(self, frame_data: List[str]):
        try:
            channel_values = list(map(float, frame_data))
        except ValueError:
            raise self.ParseError(f"Invalid values in frame data: {' '.join(frame_data)}")

        if len(channel_values) != self.__channel_count:
            raise self.ParseError(f"Expected {self.__channel_count} channel values, got {len(channel_values)} instead")

        offset = 0
        for node_name, channels in self.__node_channels:
            has_position = node_name in self.__position_data
            has_rotation = node_name in self.__rotation_data

            values = channel_values[offset : offset + len(channels)]
            position, rotation = self.__parse_channel_value(channels, values)

            if has_position:
                self.__position_data[node_name].append(position)
            if has_rotation:
                self.__rotation_data[node_name].append(rotation.as_quat())  # scalar-last

            offset += len(channels)

    def __parse_channel_value(self, channels: List[CHANNEL_TYPES], values: List[float]) -> Tuple[np.ndarray, R]:
        position = np.zeros(3)
        rot_order = ""
        rot_values = []

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

    def __get_end_site_name(self, parent_node_name: str) -> str:
        return f"{parent_node_name}_EndSite"


#     def calculate_positions(self, motion_data):
#         positions = []
#         for frame in motion_data:
#             # Assuming frame contains rotation data for each joint
#             for joint_name, rotation_data in frame.items():
#                 if len(rotation_data) == 6:
#                     offset = rotation_data[:3]
#                     self.joints[joint_name].offset = offset
#                     rotation = R.from_euler('ZYX', rotation_data[3:], degrees=True)
#                 elif len(rotation_data) == 3:
#                     rotation = R.from_euler('ZXY', rotation_data[:3], degrees=True)
#                 self.joints[joint_name].rotation = rotation

#             # Calculate transforms starting from the root
#             self.root.calculate_transform()
#             positions.append(self.get_joint_positions())
#         return positions

#     def get_joint_positions(self):
#         positions = {}
#         for name, joint in self.joints.items():
#             positions[name] = joint.get_world_position()
#         return positions
