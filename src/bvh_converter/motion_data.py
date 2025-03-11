import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List, Dict, Optional

from bvh_converter.kinematic_tree import KinematicTree


class MotionData:

    def __init__(
        self,
        kinematic_tree: KinematicTree,
        position_data: Optional[Dict[str, np.ndarray]] = None,
        rotation_data: Optional[Dict[str, np.ndarray]] = None,
        frame_time: float = 1 / 30,
    ):
        """
        A class for storing motion data.

        Attributes:
            kinematic_tree (KinematicTree): The kinematic tree that the motion data is associated with.
            position_data (Optional[Dict[str, np.ndarray]]): A dictionary of position data for each node in the kinematic tree. The keys are node names and the values are numpy arrays of shape (N, 3), where N is the number of frames.
            rotation_data (Optional[Dict[str, np.ndarray]]): A dictionary of rotation data for each node in the kinematic tree. The keys are node names and the values are numpy arrays of shape (N, 4), where N is the number of frames. The order of the quaternion elements is (x, y, z, w).
            frame_time (float): The time between frames in seconds.
        """

        frame_count = 0
        if position_data:
            frame_count = len(list(position_data.values())[0])
            for name, position in position_data.items():
                if position.shape[0] != frame_count:
                    raise ValueError(
                        f"Position data for '{name}' must have the same number of frames as other data: {position.shape[0]} frames, expected {frame_count}"
                    )
                if position.shape[1] != 3:
                    raise ValueError(f"Position data for '{name}' must have shape (N, 3): {position.shape}")

        if rotation_data:
            if frame_count == 0:
                frame_count = len(list(rotation_data.values())[0])

            for name, rotation in rotation_data.items():
                if rotation.shape[0] != frame_count:
                    raise ValueError(
                        f"Rotation data for '{name}' must have the same number of frames as other data: {rotation.shape[0]} frames, expected {frame_count}"
                    )
                if rotation.shape[1] != 4:
                    raise ValueError(f"Rotation data for '{name}' must have shape (N, 4): {rotation.shape}")

        self.kinematic_tree = kinematic_tree
        self.__position_data = position_data if position_data else {}
        self.__rotation_data = rotation_data if rotation_data else {}
        self.__frame_count = frame_count
        self.frame_time = frame_time

    @property
    def frame_count(self) -> int:
        return self.__frame_count

    # @property
    # def position_data(self) -> Dict[str, np.ndarray]:
    #     return self.__position_data.copy()

    # @property
    # def rotation_data(self) -> Dict[str, np.ndarray]:
    #     return self.__rotation_data.copy()

    def get_positions(self, name: str) -> np.ndarray:
        return self.__position_data[name]

    def get_rotations(self, name: str) -> np.ndarray:
        return self.__rotation_data[name]

    def set_positions(self, name: str, position: np.ndarray):
        if position.shape[0] != self.__frame_count:
            raise ValueError(
                f"Position data for '{name}' must have the same number of frames as other data: {position.shape[0]} frames, expected {self.__frame_count}"
            )
        if position.shape[1] != 3:
            raise ValueError(f"Position data for '{name}' must have shape (N, 3): {position.shape}")

        self.__position_data[name] = position

    def set_rotations(self, name: str, rotation: np.ndarray):
        if rotation.shape[0] != self.__frame_count:
            raise ValueError(
                f"Rotation data for '{name}' must have the same number of frames as other data: {rotation.shape[0]} frames, expected {self.__frame_count}"
            )
        if rotation.shape[1] != 4:
            raise ValueError(f"Rotation data for '{name}' must have shape (N, 4): {rotation.shape}")

        self.__rotation_data[name] = rotation
