# slam_node -> depth_node, one per keyframe.
Header header              # stamp = the exact stamp of the image the keyframe was made from
uint64 kf_id               # KeyFrame::mnId, for logs
sensor_msgs/Image image    # mono8, the image as tracked (no filtering)
