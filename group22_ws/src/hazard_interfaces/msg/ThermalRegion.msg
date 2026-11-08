# One connected hot region of a thermal image, in image coordinates:
# origin at the top-left pixel, x increasing rightward, y increasing downward,
# units of pixels.

# Bounding box, half-open: columns [bbox_x, bbox_x + bbox_width),
# rows [bbox_y, bbox_y + bbox_height).
uint32 bbox_x
uint32 bbox_y
uint32 bbox_width
uint32 bbox_height

# Unweighted mean of the region's pixel-index coordinates.
float64 centroid_x
float64 centroid_y

# Number of pixels in the region.
uint32 area_px

# Raw mono8 grey levels of the region's pixels (0-255), not temperatures.
float64 mean_intensity
uint8 max_intensity

# True if the bounding box touches an image edge; the region may be truncated.
bool touches_image_boundary
