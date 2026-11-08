# Image-space thermal evidence extracted from one successfully processed
# thermal camera frame. This is sensor evidence only: it does not state that a
# fire or hazard exists, and carries no position, distance, confidence or risk.
#
# One message is published per successfully processed frame. A frame that
# could not be processed produces no message. A missing message means that
# evidence is unavailable, not that a hazard has disappeared.

# stamp: acquisition time of the source image, copied from its header (never
# the processing time). frame_id: the source image's frame_id, copied
# unchanged ('' if unset); it does not guarantee calibrated optical-frame
# geometry for the pixel coordinates below.
std_msgs/Header header

# Source image size in pixels.
uint32 image_width
uint32 image_height

# Extraction settings that define the measurements below. A pixel is hot if
# its mono8 value is strictly greater than threshold. Regions with fewer than
# min_area_px pixels are discarded. connectivity is 4 or 8.
uint8 threshold
uint32 min_area_px
uint8 connectivity

# Every hot pixel in the image, counted before min_area_px filtering.
uint32 hot_pixel_count

# Retained regions, ordered by the raster position (row, then column) of each
# region's first pixel. The order is deterministic within a frame but is not a
# persistent identity across frames. An empty array means the frame was
# processed and no region was retained; it does not confirm that no hazard
# exists.
ThermalRegion[] regions
