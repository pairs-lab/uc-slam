#!/usr/bin/env python3
"""Converts a EuRoC or TUM-VI sequence folder (ASL format, mav0/cam0, cam1, imu0) into a ROS 2 bag.

Every message keeps the dataset stamp in nanoseconds as its header stamp, the key of the depth cache, so a replay
with depth:=file finds the cached map of each keyframe.

  ros2 run ucslam_ros asl_to_bag.py --seq data/euroc/MH_01_easy --out bags/MH_01_easy [--stereo]

Topics: /cam0/image_raw and, with --stereo, /cam1/image_raw (sensor_msgs/Image, mono8), /imu0 (sensor_msgs/Imu).
"""
import argparse
import array
import csv
import heapq
import os
import sys

import cv2
import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu


def read_csv(path):
    with open(path) as f:
        return [row for row in csv.reader(f) if row and not row[0].startswith("#")]


def stamp(msg, ns):
    msg.header.stamp.sec = ns // 1000000000
    msg.header.stamp.nanosec = ns % 1000000000


def images(seq, cam, topic):
    folder = os.path.join(seq, "mav0", cam)
    for row in read_csv(os.path.join(folder, "data.csv")):
        ns = int(row[0])
        yield ns, topic, ("image", os.path.join(folder, "data", row[1].strip()), cam)


def imu(seq, topic):
    for row in read_csv(os.path.join(seq, "mav0", "imu0", "data.csv")):
        yield int(row[0]), topic, ("imu", [float(v) for v in row[1:7]])


def to_msg(ns, item):
    if item[0] == "imu":
        w = item[1]
        m = Imu()
        stamp(m, ns)
        m.header.frame_id = "imu0"
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = w[0], w[1], w[2]
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = w[3], w[4], w[5]
        m.orientation_covariance[0] = -1.0   # no orientation estimate
        return m
    _, path, cam = item
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    m = Image()
    stamp(m, ns)
    m.header.frame_id = cam
    m.height, m.width = int(im.shape[0]), int(im.shape[1])
    m.encoding = "mono8"
    m.is_bigendian = 0
    m.step = m.width
    m.data = array.array("B", im.tobytes())   # array.array skips the per-byte checks of bytes
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="sequence folder holding mav0/")
    ap.add_argument("--out", required=True, help="bag folder to create (must not exist)")
    ap.add_argument("--stereo", action="store_true", help="also write cam1")
    ap.add_argument("--storage", default="mcap", help="rosbag2 storage plugin: mcap (default) or sqlite3")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists")

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=a.out, storage_id=a.storage), rosbag2_py.ConverterOptions("", ""))
    topics = [("/cam0/image_raw", "sensor_msgs/msg/Image"), ("/imu0", "sensor_msgs/msg/Imu")]
    if a.stereo:
        topics.append(("/cam1/image_raw", "sensor_msgs/msg/Image"))
    for name, typ in topics:
        writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr"))

    streams = [images(a.seq, "cam0", "/cam0/image_raw"), imu(a.seq, "/imu0")]
    if a.stereo:
        streams.append(images(a.seq, "cam1", "/cam1/image_raw"))
    n = {}
    for ns, topic, item in heapq.merge(*streams, key=lambda e: e[0]):   # time order across the topics
        writer.write(topic, serialize_message(to_msg(ns, item)), ns)
        n[topic] = n.get(topic, 0) + 1
    del writer   # closes the bag
    print(f"wrote {a.out}: " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())))


if __name__ == "__main__":
    main()
