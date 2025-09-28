import numpy as np
from numpy.typing import NDArray

from mocap_converter.kinematic_tree import KinematicTree
from mocap_converter.node import Node


def adjust_kinematic_tree(
    tree: KinematicTree, positions: dict[str, NDArray[np.float64]], frame: int = 0
) -> KinematicTree:
    new_nodes: list[Node] = []
    for node in tree.nodes.values():
        position = positions[node.name]
        parent = tree.get_parent(node.name)
        if parent is None:
            new_nodes.append(node.copy_with(offset=position[frame]))
            continue
        else:
            parent_position = positions[parent.name]
            offsets = position - parent_position
            offset_norms: NDArray[np.float64] = np.linalg.norm(offsets, axis=1)
            mean_norm = float(np.mean(offset_norms))

            offset = offsets[frame]
            offset_norm = offset_norms[frame]
            normalized_offset = offset / offset_norm if offset_norm > 0.0 else offset
            new_nodes.append(node.copy_with(offset=normalized_offset * mean_norm))

    return KinematicTree.from_nodes(new_nodes)
