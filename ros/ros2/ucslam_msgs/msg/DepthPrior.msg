# depth_node -> slam_node, the metric depth of one keyframe image.
uint8 VIEW_CAMERA=1
uint8 VIEW_PINHOLE=2
std_msgs/Header header       # stamp copied from the DepthRequest
string model                 # must equal UCSLAM.DepthModel of the settings
string model_source          # e.g. huggingface:<id>@<revision>
string backend               # e.g. torch-fp32; must equal the slam_node parameter depth_backend
string domain                # per-domain constants key; must equal UCSLAM.Dataset of the settings
uint8 view
uint32 full_width            # network input size
uint32 full_height
float32[4] k_view            # fx fy cx cy, VIEW_PINHOLE only
sensor_msgs/Image depth      # 32FC1 metres at map size, 0 = no prediction
float32 net_ms               # network time
float32 queue_ms             # request received to inference start
