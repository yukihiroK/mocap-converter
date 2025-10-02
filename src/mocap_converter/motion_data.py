from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from mocap_converter.kinematic_tree import KinematicTree


def _freeze_array(a: NDArray[np.float64], *, shape_second: int) -> NDArray[np.float64]:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != shape_second:
        raise ValueError(f"Array must have shape (N, {shape_second}): {arr.shape}")
    # Mark as read-only to prevent external mutation; zero-copy if already float64
    arr.setflags(write=False)
    return arr


def _freeze_mapping(
    data: Mapping[str, NDArray[np.float64]] | None, *, shape_second: int
) -> MappingProxyType[str, NDArray[np.float64]]:
    if not data:
        return MappingProxyType({})
    frozen: dict[str, NDArray[np.float64]] = {}
    for name, array in data.items():
        frozen[name] = _freeze_array(array, shape_second=shape_second)
    return MappingProxyType(frozen)


def _validate_mapping_nodes(
    tree: KinematicTree, data: Mapping[str, NDArray[np.float64]] | None
) -> dict[str, NDArray[np.float64]]:
    """Validate that all mapping keys exist in the kinematic tree.

    Returns a shallow-copied dict to decouple from the caller while keeping
    array objects for subsequent freezing step.
    """
    if not data:
        return {}
    for name in data.keys():
        if name not in tree.nodes:
            raise KeyError(f"Unknown node '{name}' not found in kinematic tree")
    return dict(data)


def _infer_and_validate_frame_count(
    pos_map: Mapping[str, NDArray[np.float64]],
    rot_map: Mapping[str, NDArray[np.float64]],
) -> int:
    """Infer frame_count from provided maps and validate consistency across nodes.

    Prefers positions as the primary source when present; falls back to rotations.
    Raises ValueError when any node's frame count mismatches the inferred value.
    """
    frame_count = 0
    if len(pos_map) > 0:
        first = next(iter(pos_map.values()))
        frame_count = first.shape[0]
        for name, arr in pos_map.items():
            if arr.shape[0] != frame_count:
                raise ValueError(f"Position data for '{name}' must have same frames: {arr.shape[0]} vs {frame_count}")
    if len(rot_map) > 0:
        if frame_count == 0:
            first = next(iter(rot_map.values()))
            frame_count = first.shape[0]
        for name, arr in rot_map.items():
            if arr.shape[0] != frame_count:
                raise ValueError(f"Rotation data for '{name}' must have same frames: {arr.shape[0]} vs {frame_count}")
    return int(frame_count)


@dataclass(frozen=True)
class MotionData:
    """Immutable carrier of motion data.

    - positions: per-node arrays of shape (frames, 3), float64
    - rotations: per-node arrays of shape (frames, 4), float64 (xyzw quaternion)
    - All arrays are marked read-only; mappings are MappingProxyType
    """

    kinematic_tree: KinematicTree
    frame_time: float = 1 / 30

    # Internal immutable state
    _positions: MappingProxyType[str, NDArray[np.float64]] = field(init=False, repr=False)
    _rotations: MappingProxyType[str, NDArray[np.float64]] = field(init=False, repr=False)
    _frame_count: int = field(init=False, repr=False)

    def __init__(
        self,
        kinematic_tree: KinematicTree,
        positions: Mapping[str, NDArray[np.float64]] | None = None,
        rotations: Mapping[str, NDArray[np.float64]] | None = None,
        frame_time: float = 1 / 30,
    ) -> None:
        object.__setattr__(self, "kinematic_tree", kinematic_tree)
        object.__setattr__(self, "frame_time", float(frame_time))

        validated_positions = _validate_mapping_nodes(kinematic_tree, positions)
        validated_rotations = _validate_mapping_nodes(kinematic_tree, rotations)

        pos_map = _freeze_mapping(validated_positions, shape_second=3)
        rot_map = _freeze_mapping(validated_rotations, shape_second=4)
        frame_count = _infer_and_validate_frame_count(pos_map, rot_map)

        object.__setattr__(self, "_positions", pos_map)
        object.__setattr__(self, "_rotations", rot_map)
        object.__setattr__(self, "_frame_count", int(frame_count))

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def has_positions(self, name: str) -> bool:
        return name in self._positions

    def has_rotations(self, name: str) -> bool:
        return name in self._rotations

    @property
    def positions(self) -> Mapping[str, NDArray[np.float64]]:
        """Read-only mapping of all node positions (frames x 3). Arrays are write-protected."""
        return self._positions

    @property
    def rotations(self) -> Mapping[str, NDArray[np.float64]]:
        """Read-only mapping of all node rotations (frames x 4, quaternion xyzw). Arrays are write-protected."""
        return self._rotations

    def copy_with(
        self,
        *,
        positions: Mapping[str, NDArray[np.float64]] | None = None,
        rotations: Mapping[str, NDArray[np.float64]] | None = None,
        frame_time: float | None = None,
    ) -> "MotionData":
        """Create a new MotionData with updated properties.

        Semantics:
        - positions=None keeps current positions; pass an empty dict to remove all positions.
        - rotations=None keeps current rotations; pass an empty dict to remove all rotations.
        - frame_time=None keeps the current frame_time.
        """
        # Fast path: no changes requested
        if positions is None and rotations is None and frame_time is None:
            return self

        # Prepare positions
        if positions is None:
            new_pos_map: Mapping[str, NDArray[np.float64]] = self._positions
        else:
            validated_positions = _validate_mapping_nodes(self.kinematic_tree, positions)
            new_pos_map = _freeze_mapping(validated_positions, shape_second=3)

        # Prepare rotations
        if rotations is None:
            new_rot_map: Mapping[str, NDArray[np.float64]] = self._rotations
        else:
            validated_rotations = _validate_mapping_nodes(self.kinematic_tree, rotations)
            new_rot_map = _freeze_mapping(validated_rotations, shape_second=4)

        # Validate combined frame count
        _ = _infer_and_validate_frame_count(new_pos_map, new_rot_map)

        new_frame_time = self.frame_time if frame_time is None else float(frame_time)

        # Construct a fresh MotionData
        return MotionData(
            self.kinematic_tree,
            positions=new_pos_map,
            rotations=new_rot_map,
            frame_time=new_frame_time,
        )
