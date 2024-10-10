import numpy as np
from typing import Optional, Set, List, TypedDict, Literal, Union


RotationOrder = Literal["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]


class _RequiredNodeParams(TypedDict):
    name: str
    parent: Optional[str]


class _OptionalNodeParams(_RequiredNodeParams, total=False):
    rotation_order: RotationOrder
    offset: np.ndarray


class Node:
    """A node in a kinematic tree, representing a joint or an end effector.

    Each node has a name, a parent, and a list of children.
    The parent and children are other nodes in the tree.

    Attributes:
        name: The name of the node.
        parent: The parent node of the node.
        children: A list of child nodes of the node.

    Methods:
        __init__: Initialize the node with a name and an optional parent.
        _add_child: Add a child node to the node.
        _remove_child: Remove a child node from the node.

    """

    NodeParams = Union[_RequiredNodeParams, _OptionalNodeParams]

    def __init__(
        self,
        name: str,
        parent: Optional["Node"] = None,
        rotation_order: Optional[RotationOrder] = None,
        offset: Optional[np.ndarray] = None,
    ):
        self.__name = name
        self.__parent = parent
        self.__children: List["Node"] = []
        self.rotation_order = rotation_order if rotation_order else "XYZ"
        self.offset = offset if offset is not None else np.zeros(3)

        if parent is not None:
            parent._add_child(self)
            self.__depth = parent.depth + 1
        else:
            self.__depth = 0

    def _add_child(self, child: "Node"):
        if child in self.__children:
            return
        self.__children.append(child)
        child.__parent = self

    def _remove_child(self, child: "Node"):
        self.__children.remove(child)
        child.__parent = None

    @property
    def name(self) -> str:
        return self.__name

    @property
    def parent(self) -> Optional["Node"]:
        return self.__parent

    @property
    def children(self) -> List["Node"]:
        return self.__children.copy()

    @property
    def depth(self) -> int:
        return self.__depth

    def __eq__(self, other):
        return self.__name == other.name

    def __repr__(self):
        return f"Node({self.__name})"

    def __str__(self):
        return self.__name
