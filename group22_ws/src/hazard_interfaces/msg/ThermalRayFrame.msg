# Camera rays for every region of one successfully processed thermal
# observation. Sensor-derived only; no simulator ground truth is used.
#
# One message is published per observation that was converted successfully.
# An empty rays array means the observation was valid and contained no
# regions. A rejected observation (invalid content, missing calibration or
# transform) produces no message at all.

# stamp: the source observation's acquisition time, copied unchanged.
# frame_id: the output frame of every origin and direction (e.g. base_link).
std_msgs/Header header

# Frame of the source observation and its calibration (e.g. thermal_link).
string source_frame_id

# Source image size in pixels.
uint32 image_width
uint32 image_height

# One ray per region, in the source observation's region order.
ThermalRegionRay[] rays
