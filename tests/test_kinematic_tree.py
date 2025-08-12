import pytest

from bvh_converter.node import Node
from bvh_converter.kinematic_tree import KinematicTree


def test_kinematic_tree_remove_node():
    root = Node("root")
    child1 = Node("child1", parent_name="root")
    child2 = Node("child2", parent_name="root")
    grandchild1 = Node("grandchild1", parent_name="child1")
    grandchild2 = Node("grandchild2", parent_name="child1")

    tree = KinematicTree.from_nodes([root, child1, child2, grandchild1, grandchild2])

    assert tree.get_node("root") == root
    assert tree.get_node("child1") == child1
    assert tree.get_node("child2") == child2
    assert tree.get_node("grandchild1") == grandchild1
    assert tree.get_node("grandchild2") == grandchild2

    # Immutable: remove_node returns a new tree
    new_tree = tree.remove_node("child1")

    # Original tree unchanged
    assert tree.get_node("root") == root
    assert tree.get_node("child1") == child1
    assert tree.get_node("child2") == child2
    assert tree.get_node("grandchild1") == grandchild1
    assert tree.get_node("grandchild2") == grandchild2

    # New tree has nodes removed
    assert new_tree.get_node("root") == root
    with pytest.raises(KeyError):
        new_tree.get_node("child1")
    assert new_tree.get_node("child2") == child2
    with pytest.raises(KeyError):
        new_tree.get_node("grandchild1")
    with pytest.raises(KeyError):
        new_tree.get_node("grandchild2")


def test_kinematic_tree_from_dict_remove_node():
    nodes: list[Node.NodeParams] = [
        {"name": "root", "parent_name": None},
        {"name": "child1", "parent_name": "root"},
        {"name": "child2", "parent_name": "root"},
        {"name": "grandchild1", "parent_name": "child1"},
        {"name": "grandchild2", "parent_name": "child1"},
    ]

    tree = KinematicTree.from_params(nodes)

    root = tree.get_node("root")
    child1 = tree.get_node("child1")
    child2 = tree.get_node("child2")
    grandchild1 = tree.get_node("grandchild1")
    grandchild2 = tree.get_node("grandchild2")

    assert root is not None
    assert child1 is not None
    assert child2 is not None
    assert grandchild1 is not None
    assert grandchild2 is not None

    assert root.parent_name is None
    assert child1.parent_name == "root"
    assert child2.parent_name == "root"
    assert grandchild1.parent_name == "child1"
    assert grandchild2.parent_name == "child1"

    # Test parent-child relationships through tree
    assert tree.get_parent("root") is None
    assert tree.get_parent("child1") == root
    assert tree.get_parent("child2") == root
    assert tree.get_parent("grandchild1") == child1
    assert tree.get_parent("grandchild2") == child1

    # Immutable: remove_node returns a new tree
    new_tree = tree.remove_node("child1")

    # New tree has nodes removed
    assert new_tree.get_node("root") == root
    with pytest.raises(KeyError):
        new_tree.get_node("child1")
    assert new_tree.get_node("child2") == child2
    with pytest.raises(KeyError):
        new_tree.get_node("grandchild1")
    with pytest.raises(KeyError):
        new_tree.get_node("grandchild2")


def test_kinematic_tree_from_dict_remove_root():
    nodes: list[Node.NodeParams] = [
        {"name": "root", "parent_name": None},
        {"name": "child1", "parent_name": "root"},
        {"name": "child2", "parent_name": "root"},
        {"name": "grandchild1", "parent_name": "child1"},
        {"name": "grandchild2", "parent_name": "child1"},
    ]

    tree = KinematicTree.from_params(nodes)

    root = tree.get_node("root")
    child1 = tree.get_node("child1")
    child2 = tree.get_node("child2")
    grandchild1 = tree.get_node("grandchild1")
    grandchild2 = tree.get_node("grandchild2")

    assert root is not None
    assert child1 is not None
    assert child2 is not None
    assert grandchild1 is not None
    assert grandchild2 is not None

    assert root.parent_name is None
    assert child1.parent_name == "root"
    assert child2.parent_name == "root"
    assert grandchild1.parent_name == "child1"
    assert grandchild2.parent_name == "child1"

    # Test parent relationships through tree
    assert tree.get_parent("root") is None
    assert tree.get_parent("child1") == root
    assert tree.get_parent("child2") == root
    assert tree.get_parent("grandchild1") == child1
    assert tree.get_parent("grandchild2") == child1

    # Immutable: remove_node returns a new tree
    new_tree = tree.remove_node("root")

    # New tree should be empty (all nodes removed when root is removed)
    with pytest.raises(KeyError):
        new_tree.get_node("root")
    with pytest.raises(KeyError):
        new_tree.get_node("child1")
    with pytest.raises(KeyError):
        new_tree.get_node("child2")
    with pytest.raises(KeyError):
        new_tree.get_node("grandchild1")
    with pytest.raises(KeyError):
        new_tree.get_node("grandchild2")


def test_kinematic_tree_add_node():
    empty_tree = KinematicTree()

    assert empty_tree.root is None
    assert empty_tree.nodes == {}

    root = Node("root")
    tree1 = empty_tree.add_node(root)

    assert tree1.root == root
    assert tree1.nodes == {"root": root}
    # Original tree unchanged
    assert empty_tree.nodes == {}

    child1 = Node("child1", parent_name="root")
    tree2 = tree1.add_node(child1)

    assert tree2.root == root
    assert tree2.nodes == {"root": root, "child1": child1}
    # Previous tree unchanged
    assert tree1.nodes == {"root": root}

    child2 = Node("child2", parent_name="root")
    tree3 = tree2.add_node(child2)

    assert tree3.root == root
    assert tree3.nodes == {"root": root, "child1": child1, "child2": child2}

    grandchild1 = Node("grandchild1", parent_name="child1")
    tree4 = tree3.add_node(grandchild1)

    assert tree4.root == root
    assert tree4.nodes == {
        "root": root,
        "child1": child1,
        "child2": child2,
        "grandchild1": grandchild1,
    }

    grandchild2 = Node("grandchild2", parent_name="child1")
    tree5 = tree4.add_node(grandchild2)

    assert tree5.root == root
    assert tree5.nodes == {
        "root": root,
        "child1": child1,
        "child2": child2,
        "grandchild1": grandchild1,
        "grandchild2": grandchild2,
    }


def test_kinematic_tree_validate_no_root():
    tree = KinematicTree()

    assert tree.root is None


def test_kinematic_tree_validate_multiple_roots():
    root1 = Node("root1")
    root2 = Node("root2")

    with pytest.raises(KinematicTree.NotConnectedError):
        KinematicTree.from_nodes([root1, root2])


def test_kinematic_tree_validate_cycle():
    # Create circular reference through parent_name (no root node)
    node1 = Node("node1", parent_name="node3")
    node2 = Node("node2", parent_name="node1")
    node3 = Node("node3", parent_name="node2")  # Creates cycle: node1 -> node3 -> node2 -> node1

    # This should raise NoRootError because no node has parent_name=None
    with pytest.raises(KinematicTree.NoRootError):
        KinematicTree.from_nodes([node1, node2, node3])


def test_kinematic_tree_validate_node_not_connected():
    root = Node("root")
    child = Node("child", parent_name="nonexistent_parent")

    with pytest.raises(KinematicTree.NotConnectedError):
        KinematicTree.from_nodes([root, child])
