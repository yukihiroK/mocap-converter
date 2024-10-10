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
    assert child.depth == 1
