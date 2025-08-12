from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from bvh_converter.node import Node


@dataclass(frozen=True)
class KinematicTree:
    """An immutable kinematic tree representing a hierarchy of joints and end effectors.

    This class uses an immutable design pattern where the tree structure is built once
    and cannot be modified. Parent-child relationships are derived from the node names
    rather than maintaining mutable references.

    Attributes:
        nodes: A mapping of node names to Node instances.

    Example:
        >>> nodes = [Node("root"), Node("child", parent_name="root")]
        >>> tree = KinematicTree.from_nodes(nodes)
        >>> new_tree = tree.add_node(Node("grandchild", parent_name="child"))
    """

    nodes: dict[str, Node] = field(default_factory=dict[str, Node])

    class NoRootError(Exception):
        """Raised when the kinematic tree has no root node."""

    class NotConnectedError(Exception):
        """Raised when a node is not connected to the tree or the tree has multiple roots."""

    class CircularReferenceError(Exception):
        """Raised when a circular reference is detected in the tree structure."""

    def __post_init__(self) -> None:
        """Validate the tree structure after initialization."""
        object.__setattr__(self, "nodes", self.nodes.copy())
        self._validate_tree_structure()

    @classmethod
    def from_nodes(cls, nodes: list[Node]) -> KinematicTree:
        """Create a KinematicTree from a list of nodes.

        Args:
            nodes: List of Node instances to include in the tree.

        Returns:
            A new KinematicTree instance.

        Raises:
            NoRootError: If no root node is found.
            NotConnectedError: If multiple root nodes exist or nodes are not connected.
            CircularReferenceError: If circular references are detected.
        """
        node_dict = {node.name: node for node in nodes}
        return cls(nodes=node_dict)

    @classmethod
    def from_params(cls, node_params: list[Node.NodeParams]) -> KinematicTree:
        """Create a KinematicTree from node parameter dictionaries.

        Args:
            node_params: List of node parameter dictionaries.

        Returns:
            A new KinematicTree instance.

        Raises:
            NoRootError: If no root node is found.
            NotConnectedError: If multiple root nodes exist.
        """
        root_params = [p for p in node_params if not p.get("parent_name")]
        if len(root_params) == 0:
            raise cls.NoRootError("No root node found in node parameters")
        if len(root_params) > 1:
            raise cls.NotConnectedError("Multiple root nodes found")

        nodes: list[Node] = []
        for params in node_params:
            offset = params.get("offset")
            if offset is None:
                offset = np.zeros(3, dtype=np.float64)

            node = Node(
                name=params["name"],
                parent_name=params.get("parent_name"),
                rotation_order=params.get("rotation_order", "XYZ"),
                offset=offset,
            )
            nodes.append(node)

        return cls.from_nodes(nodes)

    def _validate_tree_structure(self) -> None:
        """Validate the tree structure for consistency."""
        if not self.nodes:
            return

        # Find root nodes
        root_nodes = [node for node in self.nodes.values() if node.is_root]
        if len(root_nodes) == 0:
            raise self.NoRootError("No root node found in the tree")
        if len(root_nodes) > 1:
            raise self.NotConnectedError(f"Multiple root nodes found: {[n.name for n in root_nodes]}")

        # Check for orphaned nodes and circular references
        visited: set[str] = set()
        for node in self.nodes.values():
            if node.name in visited:
                continue
            self._check_path_to_root(node.name, set(), visited)

    def _check_path_to_root(self, node_name: str, path: set[str], visited: set[str]) -> None:
        """Check if a node has a valid path to root without cycles."""
        if node_name in path:
            raise self.CircularReferenceError(f"Circular reference detected involving node '{node_name}'")

        if node_name in visited:
            return

        visited.add(node_name)
        node = self.nodes.get(node_name)
        if node is None:
            raise self.NotConnectedError(f"Node '{node_name}' referenced but not found in tree")

        if node.parent_name is not None:
            if node.parent_name not in self.nodes:
                raise self.NotConnectedError(f"Parent '{node.parent_name}' of node '{node_name}' not found")
            new_path = path | {node_name}
            self._check_path_to_root(node.parent_name, new_path, visited)

    def get_node(self, name: str) -> Node:
        """Get a node by name.

        Args:
            name: The name of the node to retrieve.

        Returns:
            The Node instance.

        Raises:
            KeyError: If the node is not found.
        """
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' not found in the kinematic tree")
        return self.nodes[name]

    def get_parent(self, node_name: str) -> Node | None:
        """Get the parent of a node.

        Args:
            node_name: The name of the node.

        Returns:
            The parent Node instance, or None if the node is a root.
        """
        node = self.get_node(node_name)
        return self.nodes.get(node.parent_name) if node.parent_name else None

    def get_children(self, node_name: str) -> list[Node]:
        """Get all children of a node.

        Args:
            node_name: The name of the parent node.

        Returns:
            A list of child Node instances.
        """
        return [node for node in self.nodes.values() if node.parent_name == node_name]

    def get_siblings(self, node_name: str) -> list[Node]:
        """Get all siblings of a node.

        Args:
            node_name: The name of the node.

        Returns:
            A list of sibling Node instances.
        """
        parent_node = self.get_parent(node_name)
        if parent_node is None:
            return []
        return [node for node in self.nodes.values() if node.parent_name == parent_node.name and node.name != node_name]

    @cached_property
    def root(self) -> Node | None:
        """Get the root node of the tree."""
        root_nodes = [node for node in self.nodes.values() if node.is_root]
        return root_nodes[0] if root_nodes else None

    def has_children(self, node_name: str) -> bool:
        """Check if a node has children.

        Args:
            node_name: The name of the node to check.

        Returns:
            True if the node has children, False otherwise.
        """
        return any(node.parent_name == node_name for node in self.nodes.values())

    def is_leaf(self, node_name: str) -> bool:
        """Check if a node is a leaf (has no children).

        Args:
            node_name: The name of the node to check.

        Returns:
            True if the node is a leaf, False otherwise.
        """
        return not self.has_children(node_name)

    def has_siblings(self, node_name: str) -> bool:
        """Check if a node has siblings.

        Args:
            node_name: The name of the node to check.

        Returns:
            True if the node has siblings, False otherwise.
        """
        parent_node = self.get_parent(node_name)
        if parent_node is None:
            return False
        return any(sibling.name != node_name for sibling in self.get_children(parent_node.name))

    def get_depth(self, node_name: str) -> int:
        """Get the depth of a node in the tree.

        Args:
            node_name: The name of the node.

        Returns:
            The depth of the node (0 for root).
        """
        depth = 0
        current_name = node_name

        while True:
            node = self.get_node(current_name)
            if node.parent_name is None:
                break
            current_name = node.parent_name
            depth += 1

        return depth

    def copy_with(
        self,
        *,
        nodes: dict[str, Node] | None = None,
    ) -> KinematicTree:
        """Create a new KinematicTree with updated nodes.

        Args:
            nodes: New nodes dictionary. If None, keeps current nodes.

        Returns:
            A new KinematicTree instance with updated nodes.
        """
        new_nodes = nodes if nodes is not None else self.nodes
        return KinematicTree(nodes=new_nodes)

    def add_node(self, node: Node) -> KinematicTree:
        """Create a new tree with an additional node.

        Args:
            node: The node to add.

        Returns:
            A new KinematicTree instance with the node added.
        """
        new_nodes = self.nodes.copy()
        new_nodes[node.name] = node
        return KinematicTree(nodes=new_nodes)

    def remove_node(self, name: str) -> KinematicTree:
        """Create a new tree without a specified node and its descendants.

        Args:
            name: The name of the node to remove.

        Returns:
            A new KinematicTree instance with the node and its descendants removed.
        """
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' not found in the kinematic tree")

        # Find all nodes to remove (node and its descendants)
        nodes_to_remove = {name}
        queue = [name]

        while queue:
            current = queue.pop(0)
            for child in self.get_children(current):
                if child not in nodes_to_remove:
                    nodes_to_remove.add(child.name)
                    queue.append(child.name)

        # Create new nodes dictionary without removed nodes
        new_nodes = {n: node for n, node in self.nodes.items() if n not in nodes_to_remove}
        return KinematicTree(nodes=new_nodes)

    def __len__(self) -> int:
        """Get the number of nodes in the tree."""
        return len(self.nodes)

    def __contains__(self, name: str) -> bool:
        """Check if a node exists in the tree."""
        return name in self.nodes

    def __repr__(self) -> str:
        root_name = self.root.name if self.root else "None"
        return f"KinematicTree(root='{root_name}', nodes={len(self.nodes)})"
