from bvh_converter.io.bvh.saver import BVHSaver
from bvh_converter.kinematic_tree import KinematicTree


def test_stringify_node_hierarchy():
    kinematic_tree = KinematicTree.from_params(
        [
            {"name": "root", "parent": None},
            {"name": "child1", "parent": "root"},
            {"name": "child2", "parent": "root"},
            {"name": "child3", "parent": "child1"},
            {"name": "child4", "parent": "child1"},
            {"name": "child5", "parent": "child2"},
        ]
    )

    saver = BVHSaver(
        rotation_orders={
            "root": "ZXY",
            "child1": "ZXY",
            "child2": "ZXY",
            "child3": "ZXY",
            "child4": "ZXY",
            "child5": "ZXY",
        }
    )
    hierarchy = saver._stringify_node_hierarchy(kinematic_tree)

    expected_hierarchy = """
HIERARCHY
ROOT root
{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT child1
  {
    OFFSET 0.000000 0.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT child3
    {
      OFFSET 0.000000 0.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.000000 0.000000 0.000000
      }
    }
    JOINT child4
    {
      OFFSET 0.000000 0.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0.000000 0.000000 0.000000
      }
    }
  }
  JOINT child2
  {
    OFFSET 0.000000 0.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.000000 0.000000 0.000000
    }
  }
}
""".strip()

    assert hierarchy.strip() == expected_hierarchy
