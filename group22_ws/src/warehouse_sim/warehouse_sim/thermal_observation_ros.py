import math

from hazard_interfaces.msg import ThermalFrameObservation as ThermalFrameObservationMsg
from hazard_interfaces.msg import ThermalRegion as ThermalRegionMsg
from hazard_perception.thermal_regions import ThermalFrameObservation, ThermalRegion

NS_PER_SECOND = 1_000_000_000
INT32_MIN = -2**31
INT32_MAX = 2**31 - 1
UINT8_MAX = 2**8 - 1
UINT32_MAX = 2**32 - 1

# Largest non-negative time a builtin_interfaces/Time (int32 sec) can hold.
MAX_STAMP_NS = INT32_MAX * NS_PER_SECOND + (NS_PER_SECOND - 1)


class ObservationConversionError(ValueError):
    """An observation value cannot be represented in the ROS message."""


def stamp_to_ns(stamp) -> int:
    """
    Validate a ``builtin_interfaces/Time`` and return integer nanoseconds.

    Malformed stamps are rejected, never normalised (e.g. ``nanosec`` of 1e9 is
    not turned into an extra second). Negative times are rejected under the
    project contract. Zero is valid. Raises ``TypeError`` or ``ValueError``.
    """
    sec, nanosec = stamp.sec, stamp.nanosec
    for name, value in (('sec', sec), ('nanosec', nanosec)):
        if type(value) is not int:  # also rejects bool and NumPy integers
            raise TypeError(f'stamp.{name} must be an int, not {type(value).__name__}')
    if not INT32_MIN <= sec <= INT32_MAX:
        raise ValueError(f'stamp.sec={sec} is outside the int32 range')
    if not 0 <= nanosec < NS_PER_SECOND:
        raise ValueError(f'stamp.nanosec={nanosec} is outside [0, {NS_PER_SECOND})')
    if sec < 0:
        raise ValueError(f'stamp.sec={sec} gives a negative time')
    return sec * NS_PER_SECOND + nanosec


def observation_to_message(observation: ThermalFrameObservation) -> ThermalFrameObservationMsg:
    """
    Build a new ``hazard_interfaces/ThermalFrameObservation`` from an observation.

    Every value is checked against its ROS wire type first; a value that would
    need coercion, truncation or that is out of range raises
    ``ObservationConversionError``. The input is not modified, and every call
    returns new message objects.
    """
    if not isinstance(observation, ThermalFrameObservation):
        raise TypeError(f'expected ThermalFrameObservation, not {type(observation).__name__}')

    if observation.source_stamp_ns is None:
        raise ObservationConversionError('observation has no acquisition timestamp')
    stamp_ns = _integer('source_stamp_ns', observation.source_stamp_ns, 0, MAX_STAMP_NS)

    extraction = observation.extraction
    message = ThermalFrameObservationMsg(
        image_width=_integer('image_width', extraction.image_width, 0, UINT32_MAX),
        image_height=_integer('image_height', extraction.image_height, 0, UINT32_MAX),
        threshold=_integer('threshold', extraction.threshold, 0, UINT8_MAX),
        min_area_px=_integer('min_area_px', extraction.min_area_px, 0, UINT32_MAX),
        connectivity=_integer('connectivity', extraction.connectivity, 0, UINT8_MAX),
        hot_pixel_count=_integer('hot_pixel_count', extraction.hot_pixel_count, 0, UINT32_MAX),
        regions=[
            _region_to_message(f'regions[{index}]', region)
            for index, region in enumerate(extraction.regions)
        ],
    )
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, NS_PER_SECOND)
    frame_id = observation.source_frame_id
    message.header.frame_id = '' if frame_id is None else frame_id
    return message


def _region_to_message(name: str, region: ThermalRegion) -> ThermalRegionMsg:
    x, y, width, height = region.bbox_xywh
    centroid_x, centroid_y = region.centroid_xy
    return ThermalRegionMsg(
        bbox_x=_integer(f'{name}.bbox_x', x, 0, UINT32_MAX),
        bbox_y=_integer(f'{name}.bbox_y', y, 0, UINT32_MAX),
        bbox_width=_integer(f'{name}.bbox_width', width, 0, UINT32_MAX),
        bbox_height=_integer(f'{name}.bbox_height', height, 0, UINT32_MAX),
        centroid_x=_finite_float(f'{name}.centroid_x', centroid_x),
        centroid_y=_finite_float(f'{name}.centroid_y', centroid_y),
        area_px=_integer(f'{name}.area_px', region.area_px, 0, UINT32_MAX),
        mean_intensity=_finite_float(f'{name}.mean_intensity', region.mean_intensity),
        max_intensity=_integer(f'{name}.max_intensity', region.max_intensity, 0, UINT8_MAX),
        touches_image_boundary=_boolean(
            f'{name}.touches_image_boundary', region.touches_image_boundary,
        ),
    )


def _integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int:  # rejects bool, float and NumPy scalars
        raise ObservationConversionError(f'{name} must be an int, not {type(value).__name__}')
    if not minimum <= value <= maximum:
        raise ObservationConversionError(f'{name}={value} is outside [{minimum}, {maximum}]')
    return value


def _finite_float(name: str, value: object) -> float:
    if type(value) is not float:
        raise ObservationConversionError(f'{name} must be a float, not {type(value).__name__}')
    if not math.isfinite(value):
        raise ObservationConversionError(f'{name}={value} is not finite')
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ObservationConversionError(f'{name} must be a bool, not {type(value).__name__}')
    return value
