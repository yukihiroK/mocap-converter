from typing import Literal

NODE_TYPES = Literal["ROOT", "JOINT", "End"]

POSITION_CHANNELS = Literal["Xposition", "Yposition", "Zposition"]
ROTATION_CHANNELS = Literal["Xrotation", "Yrotation", "Zrotation"]
CHANNEL_TYPES = Literal[POSITION_CHANNELS, ROTATION_CHANNELS]

ROTATION_ORDER = Literal["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]


def validate_channel(channel: str) -> CHANNEL_TYPES:
    if channel in ("Xposition", "Yposition", "Zposition"):
        return channel
    elif channel in ("Xrotation", "Yrotation", "Zrotation"):
        return channel
    else:
        raise ValueError(f"Invalid channel: {channel}")


def filter_position_channels(
    channels: tuple[CHANNEL_TYPES, ...],
) -> tuple[POSITION_CHANNELS, ...]:
    return tuple([channel for channel in channels if channel in ("Xposition", "Yposition", "Zposition")])


def filter_rotation_channels(
    channels: tuple[CHANNEL_TYPES, ...],
) -> tuple[ROTATION_CHANNELS, ...]:
    return tuple([channel for channel in channels if channel in ("Xrotation", "Yrotation", "Zrotation")])
