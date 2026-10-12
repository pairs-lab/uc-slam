# UC-SLAM on ROS 2 Humble, x86 with an NVIDIA GPU. Usage: ros/README.md.
#   docker build -t ucslam:humble ros/docker
# The image holds only dependencies. UC-SLAM and the ROS 2 packages are built inside the container by
# ros/docker/build_ws.sh on the target machine, because UC-SLAM compiles with -march=native.
# torch 2.11.0+cu128, torchvision 0.26.0+cu128 and transformers 5.17.0 are the versions of the repository's
# environment that computed the depth cache. The cu128 wheels carry their own CUDA runtime, so the base image
# needs no CUDA toolkit, only the host driver through --gpus.
FROM ros:humble-ros-base-jammy
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 TZ=Etc/UTC \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# ROS 2 packages of the nodes, the RealSense driver, CycloneDDS, bags in MCAP
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-realsense2-camera ros-humble-rmw-cyclonedds-cpp ros-humble-cv-bridge \
      ros-humble-tf2-ros ros-humble-message-filters ros-humble-image-transport \
      ros-humble-diagnostic-msgs ros-humble-rosbag2-storage-mcap ros-humble-rosbag2-py \
      python3-colcon-common-extensions python3-venv python3-pip python3-opencv rsync \
 && rm -rf /var/lib/apt/lists/*

# UC-SLAM build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git pkg-config libeigen3-dev libopencv-dev libboost-serialization-dev libssl-dev \
      libglew-dev libgl1-mesa-dev libegl1-mesa-dev libjpeg-dev libpng-dev libepoxy-dev \
 && rm -rf /var/lib/apt/lists/*

# Pangolin v0.6, the version of the benchmark build, with the switches of ./build.sh. System.h includes
# MapDrawer.h, which includes pangolin.h, so the headers are needed even with the viewer off. v0.6 uses
# std::numeric_limits in colour.h and other files without <limits>, which g++ 11 no longer includes on the way:
# "-include limits" adds it to every C++ file.
RUN git clone https://github.com/stevenlovegrove/Pangolin.git /opt/src/Pangolin \
 && git -C /opt/src/Pangolin checkout -q v0.6 \
 && cmake -S /opt/src/Pangolin -B /opt/src/Pangolin/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/pangolin \
      -DCMAKE_CXX_FLAGS="-include limits" \
      -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_TESTS=OFF \
      -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBREALSENSE=OFF -DBUILD_PANGOLIN_LIBREALSENSE2=OFF \
      -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF -DBUILD_PANGOLIN_LIBUVC=OFF \
      -DBUILD_PANGOLIN_DEPTHSENSE=OFF -DBUILD_PANGOLIN_TELICAM=OFF -DBUILD_PANGOLIN_PLEORA=OFF \
      -DBUILD_PANGOLIN_V4L=OFF -DBUILD_PANGOLIN_LIBDC1394=OFF -DBUILD_PANGOLIN_LIBPNG=ON -DBUILD_PANGOLIN_LIBJPEG=ON \
      -DBUILD_PANGOLIN_LIBTIFF=OFF \
      -DBUILD_PANGOLIN_LIBOPENEXR=OFF -DBUILD_PANGOLIN_ZSTD=OFF -DBUILD_PANGOLIN_LZ4=OFF -DBUILD_PANGOLIN_TOON=OFF \
 && cmake --build /opt/src/Pangolin/build -j"$(nproc)" --target install \
 && rm -rf /opt/src/Pangolin/build
ENV LD_LIBRARY_PATH=/opt/pangolin/lib

# Depth node Python: the system 3.10 that Humble's rclpy is built for, in a venv that also sees rclpy.
# numpy < 2 because Humble's compiled Python modules were built against numpy 1.x; a constraints file keeps it
# there through every install. The import line is the smoke test.
SHELL ["/bin/bash", "-c"]
RUN echo "numpy<2" > /opt/pip-constraints.txt \
 && python3 -m venv --system-site-packages /opt/ucslam-venv \
 && /opt/ucslam-venv/bin/pip install --no-cache-dir -c /opt/pip-constraints.txt "numpy<2" \
 && /opt/ucslam-venv/bin/pip install --no-cache-dir -c /opt/pip-constraints.txt torch==2.11.0 torchvision==0.26.0 \
      --index-url https://download.pytorch.org/whl/cu128 \
 && /opt/ucslam-venv/bin/pip install --no-cache-dir -c /opt/pip-constraints.txt transformers==5.17.0 pyyaml scipy \
 && source /opt/ros/humble/setup.bash \
 && /opt/ucslam-venv/bin/python -c "import numpy, torch, torchvision, transformers, rclpy, cv2; \
      assert numpy.__version__.startswith('1.'), numpy.__version__; \
      print('versions', numpy.__version__, torch.__version__, torch.version.cuda, transformers.__version__, cv2.__version__)"

# The repository is mounted at /ws/ucslam (read only), the build volume at /ws/build. The model weights come
# from a mounted Hugging Face cache, so the robot needs no internet.
ENV UCSLAM_REPO=/ws/ucslam HF_HOME=/opt/hf HF_HUB_OFFLINE=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    PATH=/opt/ucslam-venv/bin:$PATH
# Interactive shells, also those of docker exec, get ROS and the workspace of build_ws.sh.
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
 && echo '[ -f /ws/build/install/setup.bash ] && source /ws/build/install/setup.bash' >> /root/.bashrc
WORKDIR /ws
CMD ["bash"]
