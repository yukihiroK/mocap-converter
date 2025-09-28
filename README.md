# mocap-converter

[![PyPI - Version](https://img.shields.io/pypi/v/mocap-converter.svg)](https://pypi.org/project/mocap-converter)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mocap-converter.svg)](https://pypi.org/project/mocap-converter)

## Installation

Install with uv:

```console
$ uv add mocap-converter
```

Or install with pip:

```console
$ pip install mocap-converter
```

## Usage

### Convert Azure Kinect Data to BVH

The snippet below loads Azure Kinect Body Tracking data exported by the
[Offline Processor](https://github.com/microsoft/Azure-Kinect-Samples/tree/master/body-tracking-samples/offline_processor),
derives joint rotations from the recorded global joint positions, and saves the
result as a BVH animation.

```python
import json

from mocap_converter.adapter.azure_kinect import (
    AZURE_KINECT_KINEMATIC_TREE,
    get_positions_from_json,
)
from mocap_converter.adjust_kinematic_tree import adjust_kinematic_tree
from mocap_converter.io.bvh.saver import save_bvh
from mocap_converter.motion_data import MotionData
from mocap_converter.pos2rot import get_rotations_from_positions


with open("path/to/kinect_capture.json") as raw:
    kinect_data = json.load(raw)

positions = get_positions_from_json(kinect_data)
# Apply optional smoothing (e.g., a median filter) here if the capture contains noisy frames.
kinematic_tree = adjust_kinematic_tree(AZURE_KINECT_KINEMATIC_TREE, positions)
root = kinematic_tree.root

positional_motion = MotionData(kinematic_tree, positions)
rotations = get_rotations_from_positions(positional_motion, current_node_name=root.name)

rotational_motion = MotionData(
    kinematic_tree,
    positions={root.name: positions[root.name]},
    rotations=rotations,
)

save_bvh(rotational_motion, "path/to/output.bvh")
```

## License

`mocap-converter` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
