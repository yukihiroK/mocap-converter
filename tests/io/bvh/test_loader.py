import pytest

from typing import Generator
import tempfile
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from bvh_converter.io.bvh.loader import BVHLoader


bvh_content = """
HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT LeftUpLeg
    {
        OFFSET 1.0 0.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftLeg
        {
            OFFSET 0.0 1.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
                OFFSET 0.0 0.0 1.0
            }
        }
    }
}
MOTION
Frames: 2
Frame Time: 0.033333
0.0 0.1 0.2 0.0 0.0 0.0 90.0 0.0 90.0 180.0 90.0 0
0.1 0.2 0.3 0.0 180.0 0.0 45.0 0.0 45.0 90.0 45.0 0

"""


@pytest.fixture
def create_bvh_file() -> Generator[Path, None, None]:
    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        f.write(bvh_content.encode())
        f.flush()
        path = Path(f.name)
        yield path

    path.unlink()


def test_bvh_loader(create_bvh_file):
    filename = create_bvh_file
    loader = BVHLoader()

    # Test motion data attributes
    motion_data = loader.load_bvh(filename)
    assert motion_data is not None
    assert motion_data.frame_time == 0.033333

    # Test kinematic tree
    kinematic_tree = motion_data.kinematic_tree
    assert kinematic_tree is not None

    # Test nodes
    hips_node = kinematic_tree.get_node("Hips")
    left_up_leg_node = kinematic_tree.get_node("LeftUpLeg")
    left_leg_node = kinematic_tree.get_node("LeftLeg")
    left_leg_end_site_node = kinematic_tree.get_node("LeftLeg_EndSite")

    assert hips_node is not None
    assert left_up_leg_node is not None
    assert left_leg_node is not None
    assert left_leg_end_site_node is not None

    # Test hierarchy
    assert hips_node.parent is None
    assert left_up_leg_node.parent == hips_node
    assert left_leg_node.parent == left_up_leg_node
    assert left_leg_end_site_node.parent == left_leg_node

    # Test offsets
    assert hips_node.offset.tolist() == [0.0, 0.0, 0.0]
    assert left_up_leg_node.offset.tolist() == [1.0, 0.0, 0.0]
    assert left_leg_node.offset.tolist() == [0.0, 1.0, 0.0]
    assert left_leg_end_site_node.offset.tolist() == [0.0, 0.0, 1.0]

    # Test hip positions
    hip_positions = motion_data.get_positions("Hips")
    assert hip_positions.shape == (2, 3)
    assert hip_positions[0].tolist() == [0.0, 0.1, 0.2]
    assert hip_positions[1].tolist() == [0.1, 0.2, 0.3]

    # Test hip rotations
    test_rot = R.from_euler("ZXY", [[0.0, 0.0, 0.0], [0.0, 180.0, 0.0]], degrees=True).as_quat()
    hip_rotations = motion_data.get_rotations("Hips")
    assert hip_rotations.shape == (2, 4)
    assert hip_rotations[0].tolist() == test_rot[0].tolist()
    assert hip_rotations[1].tolist() == test_rot[1].tolist()

    # Test left up leg rotations
    test_rot = R.from_euler("ZXY", [[90.0, 0.0, 90.0], [45.0, 0.0, 45.0]], degrees=True).as_quat()
    left_up_leg_rotations = motion_data.get_rotations("LeftUpLeg")
    assert left_up_leg_rotations.shape == (2, 4)
    assert left_up_leg_rotations[0].tolist() == test_rot[0].tolist()
    assert left_up_leg_rotations[1].tolist() == test_rot[1].tolist()

    # Test left leg rotations
    test_rot = R.from_euler("ZXY", [[180.0, 90.0, 0.0], [90.0, 45.0, 0.0]], degrees=True).as_quat()
    left_leg_rotations = motion_data.get_rotations("LeftLeg")
    assert left_leg_rotations.shape == (2, 4)
    assert left_leg_rotations[0].tolist() == test_rot[0].tolist()
    assert left_leg_rotations[1].tolist() == test_rot[1].tolist()
