from bvh_converter.node import Node


class KinematicTree:
    """A kinematic tree representing a hierarchy of joints and end effectors.

    The tree has a root node, which is the top-level joint in the hierarchy.
    Each node in the tree has a name, a parent, and a list of children.
    The parent and children are other nodes in the tree.

    Attributes:
        root: The root node of the tree.
        nodes: A dictionary mapping node names to nodes.

    Methods:
        get_node: Get a node by name.
        add_node: Add a node to the tree.
        remove_node: Remove a node from the tree.

    """

    class NoRootError(Exception):
        """Raised when the kinematic tree has no root node"""

        pass

    class NotConnectedError(Exception):
        """Raised when a node is not connected to the tree or the tree has multiple roots"""

        pass

    def __init__(self, nodes: list[Node]):
        self.__root: Node | None = None
        self.__nodes: dict[str, Node] = {}

        self.add_nodes(nodes)

    @classmethod
    def from_params(cls, nodes: list[Node.NodeParams]) -> "KinematicTree":
        root_nodes = [node for node in nodes if not node.get("parent")]
        if len(root_nodes) < 1:
            raise cls.NoRootError
        if len(root_nodes) > 1:
            raise cls.NotConnectedError
        root_params = root_nodes[0]

        result = []

        def build_tree(node_params: Node.NodeParams) -> Node:
            node = Node(**{**node_params, "parent": None})
            result.append(node)
            for child_params in nodes:
                if child_params.get("parent") == node_params["name"]:
                    child_node = build_tree(child_params)
                    node._add_child(child_node)
            return node

        build_tree(root_params)

        return cls(result)

    def get_node(self, name: str) -> Node:
        if name not in self.__nodes:
            raise KeyError(f"Node '{name}' not found in the kinematic tree.")

        return self.__nodes[name]

    def add_node(self, node: Node):
        if node.name in self.__nodes:
            return

        self.__nodes[node.name] = node

        if node.parent is None:
            if self.__root is not None:
                raise self.NotConnectedError
            self.__root = node
        elif node.parent.name not in self.__nodes:
            self.add_node(node.parent)

        if self.__root is None:  # called after all nodes are added recursively
            raise self.NoRootError

    def add_nodes(self, nodes: list[Node]):
        for node in nodes:
            self.add_node(node)

    def __remove_node(self, node: Node):
        if node.parent is not None:
            node.parent._remove_child(node)
        for child in node.children:
            self.__remove_node(child)
        del self.__nodes[node.name]

    def remove_node(self, name: str):
        node = self.__nodes[name]
        if node == self.__root:
            self.__root = None
        self.__remove_node(node)

    @property
    def root(self) -> Node | None:
        return self.__root

    @property
    def nodes(self) -> dict[str, Node]:
        return self.__nodes.copy()

    def __repr__(self):
        return f"KinematicTree(root={self.__root}, nodes={self.__nodes})"
