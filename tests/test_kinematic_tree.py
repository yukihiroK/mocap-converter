import pytest

from bvh_converter.node import Node
from bvh_converter.kinematic_tree import KinematicTree


def test_kinematic_tree_remove_node():
    root = Node("root")
    child1 = Node("child1", parent=root)
    child2 = Node("child2", parent=root)
    grandchild1 = Node("grandchild1", parent=child1)
    grandchild2 = Node("grandchild2", parent=child1)

    tree = KinematicTree([root, child1, child2, grandchild1, grandchild2])

    assert tree.get_node("root") == root
    assert tree.get_node("child1") == child1
    assert tree.get_node("child2") == child2
    assert tree.get_node("grandchild1") == grandchild1
    assert tree.get_node("grandchild2") == grandchild2

    tree.remove_node("child1")

    assert tree.get_node("root") == root
    with pytest.raises(KeyError):
        tree.get_node("child1")
    assert tree.get_node("child2") == child2
    with pytest.raises(KeyError):
        tree.get_node("grandchild1")
    with pytest.raises(KeyError):
        tree.get_node("grandchild2")


def test_kinematic_tree_from_dict_remove_node():
    nodes: list[Node.NodeParams] = [
        {"name": "root", "parent": None},
        {"name": "child1", "parent": "root"},
        {"name": "child2", "parent": "root"},
        {"name": "grandchild1", "parent": "child1"},
        {"name": "grandchild2", "parent": "child1"},
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

    assert root.parent is None
    assert child1.parent == root
    assert child2.parent == root
    assert grandchild1.parent == child1
    assert grandchild2.parent == child1

    tree.remove_node("child1")

    assert tree.get_node("root") == root
    with pytest.raises(KeyError):
        tree.get_node("child1")
    assert tree.get_node("child2") == child2
    with pytest.raises(KeyError):
        tree.get_node("grandchild1")
    with pytest.raises(KeyError):
        tree.get_node("grandchild2")


def test_kinematic_tree_from_dict_remove_root():
    nodes: list[Node.NodeParams] = [
        {"name": "root", "parent": None},
        {"name": "child1", "parent": "root"},
        {"name": "child2", "parent": "root"},
        {"name": "grandchild1", "parent": "child1"},
        {"name": "grandchild2", "parent": "child1"},
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

    assert root.parent is None
    assert child1.parent == root
    assert child2.parent == root
    assert grandchild1.parent == child1
    assert grandchild2.parent == child1

    tree.remove_node("root")

    with pytest.raises(KeyError):
        tree.get_node("root")
    with pytest.raises(KeyError):
        tree.get_node("child1")
    with pytest.raises(KeyError):
        tree.get_node("child2")
    with pytest.raises(KeyError):
        tree.get_node("grandchild1")
    with pytest.raises(KeyError):
        tree.get_node("grandchild2")


def test_kinematic_tree_add_node():
    tree = KinematicTree([])

    assert tree.root is None
    assert tree.nodes == {}

    root = Node("root")
    tree.add_node(root)

    assert tree.root == root
    assert tree.nodes == {"root": root}

    child1 = Node("child1", parent=root)
    tree.add_node(child1)

    assert tree.root == root
    assert tree.nodes == {"root": root, "child1": child1}

    child2 = Node("child2", parent=root)
    tree.add_node(child2)

    assert tree.root == root
    assert tree.nodes == {"root": root, "child1": child1, "child2": child2}

    grandchild1 = Node("grandchild1", parent=child1)
    tree.add_node(grandchild1)

    assert tree.root == root
    assert tree.nodes == {
        "root": root,
        "child1": child1,
        "child2": child2,
        "grandchild1": grandchild1,
    }

    grandchild2 = Node("grandchild2", parent=child1)
    tree.add_node(grandchild2)

    assert tree.root == root
    assert tree.nodes == {
        "root": root,
        "child1": child1,
        "child2": child2,
        "grandchild1": grandchild1,
        "grandchild2": grandchild2,
    }


def test_kinematic_tree_validate_no_root():
    tree = KinematicTree([])

    assert tree.root is None


def test_kinematic_tree_validate_multiple_roots():
    root1 = Node("root1")
    root2 = Node("root2")

    with pytest.raises(KinematicTree.NotConnectedError):
        KinematicTree([root1, root2])


def test_kinematic_tree_validate_cycle():
    root = Node("root")
    child = Node("child", parent=root)
    grandchild = Node("grandchild", parent=child)
    grandchild._add_child(root)

    with pytest.raises(KinematicTree.NoRootError):
        KinematicTree([root, child, grandchild])


def test_kinematic_tree_validate_node_not_connected():
    root = Node("root")
    child = Node("child", parent=Node("parent"))

    with pytest.raises(KinematicTree.NotConnectedError):
        KinematicTree([root, child])
