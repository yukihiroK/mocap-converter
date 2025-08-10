import numpy as np

from bvh_converter.kinematic_tree import KinematicTree


class MotionData:

    def __init__(
        self,
        kinematic_tree: KinematicTree,
        positions: dict[str, np.ndarray] | None = None,
        rotations: dict[str, np.ndarray] | None = None,
        frame_time: float = 1 / 30,
    ):
        """
        A class for storing motion data.

        Attributes:
            kinematic_tree (KinematicTree): The kinematic tree that the motion data is associated with.
            positions (Optional[Dict[str, np.ndarray]]): A dictionary of position data for each node in the kinematic tree. The keys are node names and the values are numpy arrays of shape (N, 3), where N is the number of frames.
            rotations (Optional[Dict[str, np.ndarray]]): A dictionary of rotation data for each node in the kinematic tree. The keys are node names and the values are numpy arrays of shape (N, 4), where N is the number of frames. The order of the quaternion elements is (x, y, z, w).
            frame_time (float): The time between frames in seconds.
        """

        frame_count = 0
        if positions:
            frame_count = len(list(positions.values())[0])
            for name, position in positions.items():
                if position.shape[0] != frame_count:
                    raise ValueError(
                        f"Position data for '{name}' must have the same number of frames as other data: {position.shape[0]} frames, expected {frame_count}"
                    )
                if position.shape[1] != 3:
                    raise ValueError(f"Position data for '{name}' must have shape (N, 3): {position.shape}")

        if rotations:
            if frame_count == 0:
                frame_count = len(list(rotations.values())[0])

            for name, rotation in rotations.items():
                if rotation.shape[0] != frame_count:
                    raise ValueError(
                        f"Rotation data for '{name}' must have the same number of frames as other data: {rotation.shape[0]} frames, expected {frame_count}"
                    )
                if rotation.shape[1] != 4:
                    raise ValueError(f"Rotation data for '{name}' must have shape (N, 4): {rotation.shape}")

        self.kinematic_tree = kinematic_tree
        self.__node_positions = positions if positions else {}
        self.__node_rotations = rotations if rotations else {}
        self.__frame_count = frame_count
        self.frame_time = frame_time

    @property
    def frame_count(self) -> int:
        return self.__frame_count

    def has_positions(self, name: str) -> bool:
        return name in self.__node_positions

    def has_rotations(self, name: str) -> bool:
        return name in self.__node_rotations

    def get_positions(self, name: str) -> np.ndarray:
        return self.__node_positions[name].copy()

    def get_rotations(self, name: str) -> np.ndarray:
        return self.__node_rotations[name].copy()

    def set_positions(self, name: str, position: np.ndarray):
        if position.shape[0] != self.__frame_count:
            raise ValueError(
                f"Position data for '{name}' must have the same number of frames as other data: {position.shape[0]} frames, expected {self.__frame_count}"
            )
        if position.shape[1] != 3:
            raise ValueError(f"Position data for '{name}' must have shape (N, 3): {position.shape}")

        self.__node_positions[name] = position.copy()

    def set_rotations(self, name: str, rotation: np.ndarray):
        if rotation.shape[0] != self.__frame_count:
            raise ValueError(
                f"Rotation data for '{name}' must have the same number of frames as other data: {rotation.shape[0]} frames, expected {self.__frame_count}"
            )
        if rotation.shape[1] != 4:
            raise ValueError(f"Rotation data for '{name}' must have shape (N, 4): {rotation.shape}")

        self.__node_rotations[name] = rotation.copy()
