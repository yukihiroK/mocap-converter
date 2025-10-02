import mocap_converter as mc


def test_top_level_exports() -> None:
    assert mc.Node.__name__ == "Node"
    assert mc.MotionData.__name__ == "MotionData"
    assert mc.KinematicTree.__name__ == "KinematicTree"
    assert callable(mc.adjust_kinematic_tree)
    assert callable(mc.get_rotations_from_positions)
    assert callable(mc.get_positions_from_rotations)
    assert callable(mc.save_bvh)
    assert callable(mc.load_bvh)


def test_adapter_namespace_exports() -> None:
    assert hasattr(mc, "adapter")
    assert hasattr(mc.adapter, "AZURE_KINECT_KINEMATIC_TREE")
    assert hasattr(mc.adapter, "get_positions_from_json")
