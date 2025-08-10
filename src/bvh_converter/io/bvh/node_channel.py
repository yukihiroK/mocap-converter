from dataclasses import dataclass

from bvh_converter.io.bvh.types import (
    CHANNEL_TYPES,
    POSITION_CHANNELS,
    ROTATION_CHANNELS,
    ROTATION_ORDER,
    filter_position_channels,
    filter_rotation_channels,
)


@dataclass(frozen=True)
class NodeChannel:

    name: str
    # channels: Tuple[CHANNEL_TYPES, ...]
    position_channels: tuple[POSITION_CHANNELS, ...]
    rotation_channels: tuple[ROTATION_CHANNELS, ...]
    # rotation_order: ROTATION_ORDER

    @staticmethod
    def from_channels(name: str, channels: tuple[CHANNEL_TYPES, ...]) -> "NodeChannel":
        position_channels = filter_position_channels(channels)
        rotation_channels = filter_rotation_channels(channels)
        return NodeChannel(name, position_channels, rotation_channels)

    @staticmethod
    def from_rotation_order(name: str, rotation_order: ROTATION_ORDER, has_position_channels: bool) -> "NodeChannel":
        position_channels: tuple[POSITION_CHANNELS, ...] = (
            ("Xposition", "Yposition", "Zposition") if has_position_channels else ()
        )
        rotation_channels = _get_rotation_channels_from_order(rotation_order)
        return NodeChannel(name, position_channels, rotation_channels)

    @property
    def channels(self) -> tuple[CHANNEL_TYPES, ...]:
        return self.position_channels + self.rotation_channels  # position channels first

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def has_position_channels(self) -> bool:
        return bool(self.position_channels)

    @property
    def has_rotation_channels(self) -> bool:
        return bool(self.rotation_channels)

    @property
    def rotation_order(self) -> ROTATION_ORDER:
        return _get_rotation_order_from_channels(self.rotation_channels)


def _get_rotation_channels_from_order(
    rotation_order: ROTATION_ORDER,
) -> tuple[ROTATION_CHANNELS, ...]:

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


def _get_rotation_order_from_channels(
    rotation_channels: tuple[ROTATION_CHANNELS, ...],
) -> ROTATION_ORDER:

    if rotation_channels == ("Xrotation", "Yrotation", "Zrotation"):
        return "XYZ"
    elif rotation_channels == ("Xrotation", "Zrotation", "Yrotation"):
        return "XZY"
    elif rotation_channels == ("Yrotation", "Xrotation", "Zrotation"):
        return "YXZ"
    elif rotation_channels == ("Yrotation", "Zrotation", "Xrotation"):
        return "YZX"
    elif rotation_channels == ("Zrotation", "Xrotation", "Yrotation"):
        return "ZXY"
    elif rotation_channels == ("Zrotation", "Yrotation", "Xrotation"):
        return "ZYX"

    raise ValueError(f"Invalid rotation channels: {rotation_channels}")
