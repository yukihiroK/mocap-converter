from typing import Literal, Union, NamedTuple


NODE_TYPES = Literal["ROOT", "JOINT", "End"]

POSITION_CHANNELS = Literal["Xposition", "Yposition", "Zposition"]
ROTATION_CHANNELS = Literal["Xrotation", "Yrotation", "Zrotation"]
CHANNEL_TYPES = Union[POSITION_CHANNELS, ROTATION_CHANNELS]


def validate_channel(channel: str) -> CHANNEL_TYPES:
    if channel == "Xposition" or channel == "Yposition" or channel == "Zposition":
        return channel
    elif channel == "Xrotation" or channel == "Yrotation" or channel == "Zrotation":
        return channel
    else:
        raise ValueError(f"Invalid channel: {channel}")


class NodeChannel(NamedTuple):
    name: str
    channels: list[CHANNEL_TYPES]
