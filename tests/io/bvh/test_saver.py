# pyright: reportPrivateUsage=false

from bvh_converter.io.bvh.saver import _build_hierarchy_string
from bvh_converter.io.bvh.types import ROTATION_ORDER
from bvh_converter.kinematic_tree import KinematicTree


def test_build_hierarchy_string():
    kinematic_tree = KinematicTree.from_params(
        [
            {"name": "root", "parent_name": None},
            {"name": "child1", "parent_name": "root"},
            {"name": "child2", "parent_name": "root"},
            {"name": "child3", "parent_name": "child1"},
            {"name": "child4", "parent_name": "child1"},
            {"name": "child5", "parent_name": "child2"},
        ]
    )

    rotation_orders: dict[str, ROTATION_ORDER] = {
        "root": "ZXY",
        "child1": "ZXY",
        "child2": "ZXY",
        "child3": "ZXY",
        "child4": "ZXY",
        "child5": "ZXY",
    }
    hierarchy, _ = _build_hierarchy_string(kinematic_tree, rotation_orders)

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
