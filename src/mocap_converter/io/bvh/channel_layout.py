from dataclasses import dataclass

from mocap_converter.io.bvh.types import (
    CHANNEL_TYPES,
    POSITION_CHANNELS,
    ROTATION_CHANNELS,
    ROTATION_ORDER,
    filter_position_channels,
    filter_rotation_channels,
    rotation_channels_from_order,
    rotation_order_from_channels,
)


@dataclass(frozen=True)
class BVHChannelLayout:
    position_channels: tuple[POSITION_CHANNELS, ...]
    rotation_channels: tuple[ROTATION_CHANNELS, ...]

    @classmethod
    def from_bvh_channels(cls, channels: tuple[CHANNEL_TYPES, ...]) -> "BVHChannelLayout":
        position_channels = filter_position_channels(channels)
        rotation_channels = filter_rotation_channels(channels)
        return cls(position_channels, rotation_channels)

    @classmethod
    def from_rotation_order(cls, rotation_order: ROTATION_ORDER, has_position_channels: bool) -> "BVHChannelLayout":
        position_channels: tuple[POSITION_CHANNELS, ...] = (
            ("Xposition", "Yposition", "Zposition") if has_position_channels else ()
        )
        rotation_channels = rotation_channels_from_order(rotation_order)
        return cls(position_channels, rotation_channels)

    @property
    def channels(self) -> tuple[CHANNEL_TYPES, ...]:
        return self.position_channels + self.rotation_channels

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
        return rotation_order_from_channels(self.rotation_channels)
