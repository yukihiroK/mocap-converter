from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

RotationOrder = Literal["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]


class _RequiredNodeParams(TypedDict):
    name: str
    parent_name: str | None


class _OptionalNodeParams(_RequiredNodeParams, total=False):
    # rotation_order: RotationOrder
    offset: NDArray[np.float64]


class _KeepCurrentType:
    """Sentinel type for representing 'keep current value' in copy_with method."""

    def __repr__(self) -> str:
        return "_KEEP_CURRENT"


_KEEP_CURRENT: Final = _KeepCurrentType()


@dataclass(frozen=True)
class Node:
    """An immutable node in a kinematic tree, representing a joint or an end effector.

    This class uses an immutable design pattern where any modifications return a new
    instance rather than modifying the existing one. Parent-child relationships are
    managed by name references rather than direct object references to avoid circular
    references and maintain immutability.

    Attributes:
        name: The unique name of the node.
        parent_name: The name of the parent node, or None if this is a root node.
        rotation_order: The order of rotation axes (default: "XYZ").
        offset: The 3D offset from the parent node (default: zero vector).

    Methods:
        copy_with: Create a new node with updated properties.

    Example:
        >>> root = Node(name="root", parent_name=None)
        >>> child = Node(name="child", parent_name="root")
        >>> updated_child = child.copy_with(rotation_order="ZYX")
    """

    name: str
    parent_name: str | None = None
    # rotation_order: RotationOrder = "XYZ"
    offset: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    # Type alias for backward compatibility
    NodeParams = _RequiredNodeParams | _OptionalNodeParams

    def __post_init__(self) -> None:
        # Ensure offset is a defensive copy and has correct shape
        if self.offset.shape != (3,):
            raise ValueError(f"offset must have shape (3,), got {self.offset.shape}")
        # Create a defensive copy to prevent external mutation
        object.__setattr__(self, "offset", self.offset.copy())

    @property
    def is_root(self) -> bool:
        """True if this node has no parent."""
        return self.parent_name is None

    def copy_with(
        self,
        *,
        name: str | None = None,
        parent_name: str | None | _KeepCurrentType = _KEEP_CURRENT,
        # rotation_order: RotationOrder | None = None,
        offset: NDArray[np.float64] | None = None,
    ) -> Node:
        """Create a new Node with updated properties.

        Args:
            name: New name for the node. If None, keeps the current name.
            parent_name: New parent name. Use None to make it a root node,
                        or _KEEP_CURRENT to keep current value.
            rotation_order: New rotation order. If None, keeps current value.
            offset: New offset vector. If None, keeps current value.

        Returns:
            A new Node instance with the updated properties.

        Example:
            >>> node = Node("joint1", "root")
            >>> updated = node.copy_with(rotation_order="ZYX", parent_name=None)
        """
        new_name = name if name is not None else self.name
        new_parent_name = self.parent_name if isinstance(parent_name, _KeepCurrentType) else parent_name
        # new_rotation_order = rotation_order if rotation_order is not None else self.rotation_order

        if offset is not None:
            if offset.shape != (3,):
                raise ValueError(f"offset must have shape (3,), got {offset.shape}")
            new_offset = offset.copy()
        else:
            new_offset = self.offset.copy()  # self.offset is guaranteed to be NDArray after __post_init__

        return Node(
            name=new_name,
            parent_name=new_parent_name,
            # rotation_order=new_rotation_order,
            offset=new_offset,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return (
            self.name == other.name
            and self.parent_name == other.parent_name
            # and self.rotation_order == other.rotation_order
            and np.array_equal(self.offset, other.offset)  # Both are NDArray after __post_init__
        )

    def __hash__(self) -> int:
        # return hash((self.name, self.parent_name, self.rotation_order, tuple(self.offset)))
        return hash((self.name, self.parent_name, tuple(self.offset)))  # self.offset is NDArray after __post_init__

    def __repr__(self) -> str:
        return f"Node(name='{self.name}', parent_name={self.parent_name!r})"

    def __str__(self) -> str:
        return self.name
