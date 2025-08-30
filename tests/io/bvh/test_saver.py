# pyright: reportPrivateUsage=false

from bvh_converter.io.bvh.saver import _build_hierarchy_string
from bvh_converter.io.bvh.channel_layout import BVHChannelLayout
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

    # Provide explicit channel layouts: root has positions; others rotations only
    channel_layouts = {
        name: BVHChannelLayout.from_rotation_order("ZXY", has_position_channels=(name == "root"))
        for name in ("root", "child1", "child2", "child3", "child4", "child5")
    }
    hierarchy, _ = _build_hierarchy_string(kinematic_tree, channel_layouts)

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
