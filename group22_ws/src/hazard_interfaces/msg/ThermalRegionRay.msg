# The camera ray through one thermal region's centroid, in the frame of the
# enclosing ThermalRayFrame's header. A ray is a bearing from the camera only:
# no range, map position, hazard identity, probability or covariance.

# Index of the region in the source ThermalFrameObservation (same stamp). The
# index is only a position within that frame, not a persistent identity.
uint32 region_index

# Region centroid in source-image pixel coordinates (pixel centres at
# integers), and whether the region touches an image edge (it, and so its
# centroid, may be truncated).
float64 centroid_x
float64 centroid_y
bool touches_image_boundary

# Camera position (metres) and unit-length direction, both in header.frame_id.
geometry_msgs/Point origin
geometry_msgs/Vector3 direction
