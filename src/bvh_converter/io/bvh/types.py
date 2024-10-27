from dataclasses import dataclass
from typing import Literal, Tuple


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


def filter_position_channels(channels: Tuple[CHANNEL_TYPES, ...]) -> Tuple[POSITION_CHANNELS, ...]:
    return tuple([channel for channel in channels if channel in ("Xposition", "Yposition", "Zposition")])


def filter_rotation_channels(channels: Tuple[CHANNEL_TYPES, ...]) -> Tuple[ROTATION_CHANNELS, ...]:
    return tuple([channel for channel in channels if channel in ("Xrotation", "Yrotation", "Zrotation")])


@dataclass(frozen=True)
class NodeChannel:

    name: str
    channels: Tuple[CHANNEL_TYPES, ...]
    position_channels: Tuple[POSITION_CHANNELS, ...]
    rotation_channels: Tuple[ROTATION_CHANNELS, ...]
    rotation_order: ROTATION_ORDER

    @staticmethod
    def from_channels(name: str, channels: Tuple[CHANNEL_TYPES, ...]) -> "NodeChannel":
        position_channels = filter_position_channels(channels)
        rotation_channels = filter_rotation_channels(channels)
        return NodeChannel(name, channels, position_channels, rotation_channels, "ZXY")

    @staticmethod
    def from_rotation_order(name: str, rotation_order: ROTATION_ORDER, has_position_channels: bool) -> "NodeChannel":
        position_channels: Tuple[POSITION_CHANNELS, ...] = (
            ("Xposition", "Yposition", "Zposition") if has_position_channels else ()
        )
        rotation_channels = _get_node_channels(rotation_order)
        channels: Tuple[CHANNEL_TYPES, ...] = position_channels + rotation_channels  # Position channels first
        return NodeChannel(name, channels, position_channels, rotation_channels, rotation_order)

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def has_position_channels(self) -> bool:
        return bool(self.position_channels)

    @property
    def has_rotation_channels(self) -> bool:
        return bool(self.rotation_channels)


def _get_node_channels(
    rotation_order: ROTATION_ORDER,
) -> Tuple[ROTATION_CHANNELS, ...]:

    if rotation_order == "XYZ":
        return "Xrotation", "Yrotation", "Zrotation"
    elif rotation_order == "XZY":
        return "Xrotation", "Zrotation", "Yrotation"
    elif rotation_order == "YXZ":
        return "Yrotation", "Xrotation", "Zrotation"
    elif rotation_order == "YZX":
        return "Yrotation", "Zrotation", "Xrotation"
    elif rotation_order == "ZXY":
        return "Zrotation", "Xrotation", "Yrotation"
    elif rotation_order == "ZYX":
        return "Zrotation", "Yrotation", "Xrotation"
