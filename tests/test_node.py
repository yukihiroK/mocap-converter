from bvh_converter.kinematic_tree import KinematicTree
from bvh_converter.node import Node


def test_node():
    parent = Node("parent")
    child = Node("child", parent_name="parent")

    # Test immutable properties
    assert child.name == "child"
    assert child.parent_name == "parent"
    assert parent.is_root
    assert not child.is_root

    # Test copy_with functionality
    updated_child = child.copy_with(parent_name=None)
    assert updated_child.name == "child"
    assert updated_child.is_root
    assert child.parent_name == "parent"  # Original unchanged


def test_node_depth():
    # Create nodes for tree structure
    root = Node("root")
    child1 = Node("child1", parent_name="root")
    child2 = Node("child2", parent_name="child1")
    child3 = Node("child3", parent_name="child2")

    # Create tree from nodes
    tree = KinematicTree.from_nodes([root, child1, child2, child3])

    # Test depth calculation through tree
    assert tree.get_depth("root") == 0
    assert tree.get_depth("child1") == 1
    assert tree.get_depth("child2") == 2
    assert tree.get_depth("child3") == 3

    # Test parent-child relationships through tree
    assert tree.get_parent("root") is None
    assert tree.get_parent("child1") == root
    assert tree.get_parent("child2") == child1
    assert tree.get_parent("child3") == child2

    # Test children relationships
    assert tree.get_children("root") == [child1]
    assert tree.get_children("child1") == [child2]
    assert tree.get_children("child2") == [child3]
    assert tree.get_children("child3") == []
