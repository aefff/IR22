from dataclasses import dataclass

import cv2
import numpy as np

_VALID_CONNECTIVITIES = (4, 8)
# A uint8 pixel can never exceed 255, so a threshold of 255 could never detect
# anything; it is rejected as a configuration error.
_MAX_THRESHOLD = 254


@dataclass(frozen=True)
class ThermalRegion:
    """
    One connected component of hot pixels in a single image.

    All values are in image space. The region carries no physical hazard
    interpretation, confidence, distance or identity.
    """

    bbox_xywh: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_px: int
    mean_intensity: float
    max_intensity: int
    touches_image_boundary: bool


@dataclass(frozen=True)
class ThermalExtractionResult:

    image_width: int
    image_height: int
    threshold: int
    min_area_px: int
    connectivity: int
    hot_pixel_count: int
    regions: tuple[ThermalRegion, ...]


@dataclass(frozen=True)
class ThermalFrameObservation:

    source_stamp_ns: int | None
    source_frame_id: str | None
    extraction: ThermalExtractionResult

    def __post_init__(self) -> None:
        stamp = self.source_stamp_ns
        if stamp is not None:
            if isinstance(stamp, bool) or not isinstance(stamp, int):
                raise TypeError(
                    'source_stamp_ns must be an int number of nanoseconds or None, '
                    f'got {type(stamp).__name__}',
                )
            if stamp < 0:
                raise ValueError(f'source_stamp_ns must be non-negative, got {stamp}')

        frame_id = self.source_frame_id
        if frame_id is not None:
            if not isinstance(frame_id, str):
                raise TypeError(
                    f'source_frame_id must be a str or None, got {type(frame_id).__name__}',
                )
            if not frame_id:
                raise ValueError('source_frame_id must not be empty; use None when unknown')

        if not isinstance(self.extraction, ThermalExtractionResult):
            raise TypeError(
                'extraction must be a ThermalExtractionResult, '
                f'got {type(self.extraction).__name__}',
            )


def extract_thermal_regions(
    image: np.ndarray,
    *,
    threshold: int = 180,
    min_area_px: int = 1,
    connectivity: int = 8,
) -> ThermalExtractionResult:

    _validate_image(image)
    threshold = _require_int('threshold', threshold)
    if not 0 <= threshold <= _MAX_THRESHOLD:
        raise ValueError(f'threshold must be in [0, {_MAX_THRESHOLD}], got {threshold}')
    min_area_px = _require_int('min_area_px', min_area_px)
    if min_area_px < 1:
        raise ValueError(f'min_area_px must be at least 1, got {min_area_px}')
    connectivity = _require_int('connectivity', connectivity)
    if connectivity not in _VALID_CONNECTIVITIES:
        raise ValueError(f'connectivity must be 4 or 8, got {connectivity}')

    height, width = image.shape
    # A new array: the comparison never writes to ``image``.
    hot_mask = image > threshold
    hot_pixel_count = int(np.count_nonzero(hot_mask))

    regions: tuple[ThermalRegion, ...] = ()
    if hot_pixel_count > 0:
        regions = _measure_regions(image, hot_mask, min_area_px, connectivity)

    return ThermalExtractionResult(
        image_width=int(width),
        image_height=int(height),
        threshold=threshold,
        min_area_px=min_area_px,
        connectivity=connectivity,
        hot_pixel_count=hot_pixel_count,
        regions=regions,
    )


def _measure_regions(
    image: np.ndarray,
    hot_mask: np.ndarray,
    min_area_px: int,
    connectivity: int,
) -> tuple[ThermalRegion, ...]:
    height, width = image.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        hot_mask.astype(np.uint8),
        connectivity=connectivity,
        ltype=cv2.CV_32S,
    )

    # Boolean indexing visits pixels in row-major (raster) order, so these
    # arrays hold exactly the hot pixels, in raster order. Background label 0
    # never appears among them.
    hot_labels = labels[hot_mask]
    hot_values = image[hot_mask]

    # Component-only intensity statistics.
    intensity_sums = np.bincount(hot_labels, weights=hot_values, minlength=num_labels)
    intensity_maxima = np.zeros(num_labels, dtype=np.uint8)
    np.maximum.at(intensity_maxima, hot_labels, hot_values)

    # Order components by the raster position of their first pixel. These
    # positions are unique per component, so the order has no ties.
    present_labels, first_positions = np.unique(hot_labels, return_index=True)
    ordered_labels = present_labels[np.argsort(first_positions)]

    regions = []
    for label in ordered_labels:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_boundary = x == 0 or y == 0 or x + w == width or y + h == height
        regions.append(ThermalRegion(
            bbox_xywh=(x, y, w, h),
            centroid_xy=(float(centroids[label, 0]), float(centroids[label, 1])),
            area_px=area,
            mean_intensity=float(intensity_sums[label]) / area,
            max_intensity=int(intensity_maxima[label]),
            touches_image_boundary=touches_boundary,
        ))
    return tuple(regions)


def _validate_image(image: object) -> None:
    # Subclasses such as masked arrays or np.matrix change indexing semantics,
    # so only plain ndarrays are accepted.
    if type(image) is not np.ndarray:
        raise TypeError(f'image must be a numpy.ndarray, got {type(image).__name__}')
    if image.ndim != 2:
        raise ValueError(
            f'image must be two-dimensional (height, width), got shape {image.shape}',
        )
    if image.size == 0:
        raise ValueError(f'image must not be empty, got shape {image.shape}')
    if image.dtype != np.uint8:
        raise TypeError(f'image dtype must be uint8, got {image.dtype}')


def _require_int(name: str, value: object) -> int:
    # bool is a subclass of int but is never a meaningful parameter value here.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer, got {type(value).__name__}')
    return int(value)
