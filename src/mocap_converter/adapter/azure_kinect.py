import numpy as np
from numpy.typing import NDArray
from typing import Any

from mocap_converter.kinematic_tree import KinematicTree

AZURE_KINECT_KINEMATIC_TREE = KinematicTree.from_params(
    [
        {"name": "PELVIS", "parent_name": None},
        {"name": "SPINE_NAVEL", "parent_name": "PELVIS"},
        {"name": "SPINE_CHEST", "parent_name": "SPINE_NAVEL"},
        {"name": "NECK", "parent_name": "SPINE_CHEST"},
        {"name": "HEAD", "parent_name": "NECK"},
        {"name": "CLAVICLE_LEFT", "parent_name": "SPINE_CHEST"},
        {"name": "SHOULDER_LEFT", "parent_name": "CLAVICLE_LEFT"},
        {"name": "ELBOW_LEFT", "parent_name": "SHOULDER_LEFT"},
        {"name": "WRIST_LEFT", "parent_name": "ELBOW_LEFT"},
        {"name": "HAND_LEFT", "parent_name": "WRIST_LEFT"},
        {"name": "HANDTIP_LEFT", "parent_name": "HAND_LEFT"},
        {"name": "THUMB_LEFT", "parent_name": "WRIST_LEFT"},
        {"name": "HIP_LEFT", "parent_name": "PELVIS"},
        {"name": "KNEE_LEFT", "parent_name": "HIP_LEFT"},
        {"name": "ANKLE_LEFT", "parent_name": "KNEE_LEFT"},
        {"name": "FOOT_LEFT", "parent_name": "ANKLE_LEFT"},
        {"name": "CLAVICLE_RIGHT", "parent_name": "SPINE_CHEST"},
        {"name": "SHOULDER_RIGHT", "parent_name": "CLAVICLE_RIGHT"},
        {"name": "ELBOW_RIGHT", "parent_name": "SHOULDER_RIGHT"},
        {"name": "WRIST_RIGHT", "parent_name": "ELBOW_RIGHT"},
        {"name": "HAND_RIGHT", "parent_name": "WRIST_RIGHT"},
        {"name": "HANDTIP_RIGHT", "parent_name": "HAND_RIGHT"},
        {"name": "THUMB_RIGHT", "parent_name": "WRIST_RIGHT"},
        {"name": "HIP_RIGHT", "parent_name": "PELVIS"},
        {"name": "KNEE_RIGHT", "parent_name": "HIP_RIGHT"},
        {"name": "ANKLE_RIGHT", "parent_name": "KNEE_RIGHT"},
        {"name": "FOOT_RIGHT", "parent_name": "ANKLE_RIGHT"},
        {"name": "NOSE", "parent_name": "HEAD"},
        {"name": "EYE_LEFT", "parent_name": "HEAD"},
        {"name": "EAR_LEFT", "parent_name": "HEAD"},
        {"name": "EYE_RIGHT", "parent_name": "HEAD"},
        {"name": "EAR_RIGHT", "parent_name": "HEAD"},
    ]
)


def get_positions_from_json(json_data: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    assert "frames" in json_data
    frames: list[dict[str, Any]] = json_data["frames"]

    positions: list[NDArray[np.float64]] = []
    end_frame = None
    for frame in frames:
        if frame["num_bodies"] == 0:
            if len(positions) > 0:
                positions.append(positions[-1])
            continue

        if frame["num_bodies"] > 1 and len(positions) > 0:
            prepositions = np.array(positions[-1])
            distances = [np.linalg.norm(prepositions - np.array(body["joint_positions"])) for body in frame["bodies"]]
            idx = np.argmin(distances)
            positions.append(frame["bodies"][idx]["joint_positions"])
        else:
            positions.append(frame["bodies"][0]["joint_positions"])
        end_frame = len(positions)

    if end_frame is not None:
        positions = positions[: end_frame + 1]
    positions_np = np.array(positions) / 10  # mm -> cm
    positions_np[:, :, 1] *= -1  # flip y axis
    positions_np[:, :, 2] *= -1  # flip z axis
    positions_np -= positions_np[0:1, 0:1]  # Move root joint of first frame to origin

    node_names: list[str] = json_data["joint_names"]
    return {name: positions_np[:, i] for i, name in enumerate(node_names)}
