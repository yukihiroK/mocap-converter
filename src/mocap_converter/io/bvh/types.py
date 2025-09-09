from typing import Literal

# Typed aliases for BVH constructs

NODE_TYPES = Literal["ROOT", "JOINT", "End"]

POSITION_CHANNELS = Literal["Xposition", "Yposition", "Zposition"]
ROTATION_CHANNELS = Literal["Xrotation", "Yrotation", "Zrotation"]
CHANNEL_TYPES = Literal[POSITION_CHANNELS, ROTATION_CHANNELS]

ROTATION_ORDER = Literal["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]

# === Rotation channel/order mappings (single source of truth) ===
# Keep these mappings centralized to avoid scattering if/elif logic

_ROTATION_CHANNELS_BY_ORDER: dict[ROTATION_ORDER, tuple[ROTATION_CHANNELS, ...]] = {
    "XYZ": ("Xrotation", "Yrotation", "Zrotation"),
    "XZY": ("Xrotation", "Zrotation", "Yrotation"),
    "YXZ": ("Yrotation", "Xrotation", "Zrotation"),
    "YZX": ("Yrotation", "Zrotation", "Xrotation"),
    "ZXY": ("Zrotation", "Xrotation", "Yrotation"),
    "ZYX": ("Zrotation", "Yrotation", "Xrotation"),
}

_ROTATION_ORDER_BY_CHANNELS: dict[tuple[ROTATION_CHANNELS, ...], ROTATION_ORDER] = {
    v: k for k, v in _ROTATION_CHANNELS_BY_ORDER.items()
}


def rotation_channels_from_order(order: ROTATION_ORDER) -> tuple[ROTATION_CHANNELS, ...]:
    return _ROTATION_CHANNELS_BY_ORDER[order]


def rotation_order_from_channels(chs: tuple[ROTATION_CHANNELS, ...]) -> ROTATION_ORDER:
    try:
        return _ROTATION_ORDER_BY_CHANNELS[chs]
    except KeyError:
        raise ValueError(f"Invalid rotation channels: {chs}")


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
    return tuple(ch for ch in channels if ch in ("Xposition", "Yposition", "Zposition"))


def filter_rotation_channels(
    channels: tuple[CHANNEL_TYPES, ...],
) -> tuple[ROTATION_CHANNELS, ...]:
    return tuple(ch for ch in channels if ch in ("Xrotation", "Yrotation", "Zrotation"))

