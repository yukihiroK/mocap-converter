from bvh_converter.node import Node


def test_node():
    parent = Node("parent")
    child = Node("child", parent=parent)

    assert child.parent == parent
    assert parent.children == [child]
    assert parent.depth == 0
    assert child.depth == 1

    parent._remove_child(child)

    assert child.parent is None
    assert parent.children == []
    assert parent.depth == 0
    assert child.depth == 0


def test_node_depth():
    root = Node("root")
    child1 = Node("child1", parent=root)
    child2 = Node("child2", parent=child1)
    child3 = Node("child3", parent=child2)

    assert root.depth == 0
    assert child1.depth == 1
    assert child2.depth == 2
    assert child3.depth == 3
