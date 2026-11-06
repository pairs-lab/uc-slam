cmake_minimum_required(VERSION 3.8)
project(ucslam_ros)

# UC-SLAM is built first by ./build.sh at the repository root. This package links its library. The repository is
# found from the real path of the package, which may be a symlink in a workspace.
get_filename_component(UCSLAM_PKG_REAL "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(UCSLAM_REPO_DEFAULT "${UCSLAM_PKG_REAL}/../../.." ABSOLUTE)
set(UCSLAM_REPO "${UCSLAM_REPO_DEFAULT}" CACHE PATH "UC-SLAM repository root")
set(UCSLAM_ROOT "${UCSLAM_REPO}/cpp/ucslam" CACHE PATH "UC-SLAM source and build tree")
set(PANGOLIN_ROOT "${UCSLAM_REPO}/third_party/Pangolin/install" CACHE PATH "Pangolin install of ./build.sh")

# rclcpp needs C++17. The library keeps its own flags. Same -march as the library, so that Eigen objects have
# the same alignment on both sides.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(ucslam_msgs REQUIRED)

# The node must use the OpenCV that libORB_SLAM3 was built with: cv::Mat crosses the library boundary, and two
# OpenCV versions in one process break it. The library build records its OpenCV in its CMakeCache.txt.
set(UCSLAM_LIB_CACHE "${UCSLAM_ROOT}/build/CMakeCache.txt")
set(UCSLAM_LIB_OPENCV_DIR "")
set(UCSLAM_LIB_OPENCV_VERSION "")
if(EXISTS "${UCSLAM_LIB_CACHE}")
  file(STRINGS "${UCSLAM_LIB_CACHE}" _ocv_dir REGEX "^OpenCV_DIR:[A-Z]+=")
  string(REGEX REPLACE "^OpenCV_DIR:[A-Z]+=" "" UCSLAM_LIB_OPENCV_DIR "${_ocv_dir}")
  # The version find_package reported for the library, e.g. [/usr][v4.5.4(4.2)]
  file(STRINGS "${UCSLAM_LIB_CACHE}" _ocv_found REGEX "^FIND_PACKAGE_MESSAGE_DETAILS_OpenCV:INTERNAL=")
  if(_ocv_found MATCHES "\\]\\[v([0-9][0-9.]*)")
    set(UCSLAM_LIB_OPENCV_VERSION "${CMAKE_MATCH_1}")
  endif()
endif()
if(UCSLAM_LIB_OPENCV_DIR)
  if(NOT EXISTS "${UCSLAM_LIB_OPENCV_DIR}")
    message(FATAL_ERROR "libORB_SLAM3 was built with OpenCV_DIR=${UCSLAM_LIB_OPENCV_DIR} (${UCSLAM_LIB_CACHE}), "
                        "which does not exist here. Rebuild UC-SLAM with ./build.sh first.")
  endif()
  set(OpenCV_DIR "${UCSLAM_LIB_OPENCV_DIR}" CACHE PATH "OpenCV of libORB_SLAM3, read from ${UCSLAM_LIB_CACHE}" FORCE)
  find_package(OpenCV 4 REQUIRED PATHS "${UCSLAM_LIB_OPENCV_DIR}" NO_DEFAULT_PATH)
  if(UCSLAM_LIB_OPENCV_VERSION AND NOT OpenCV_VERSION VERSION_EQUAL UCSLAM_LIB_OPENCV_VERSION)
    message(FATAL_ERROR "OpenCV mismatch: libORB_SLAM3 was built with OpenCV ${UCSLAM_LIB_OPENCV_VERSION} "
                        "(${UCSLAM_LIB_CACHE}) but ${OpenCV_DIR} gives ${OpenCV_VERSION}. "
                        "Rebuild UC-SLAM and this package against the same OpenCV.")
  endif()
else()
  message(WARNING "No OpenCV_DIR in ${UCSLAM_LIB_CACHE}: the OpenCV of libORB_SLAM3 is not checked. "
                  "Check with: ldd slam_node | grep libopencv_core (one line).")
  find_package(OpenCV 4 REQUIRED)
endif()
message(STATUS "slam_node uses OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR} "
               "(libORB_SLAM3: ${UCSLAM_LIB_OPENCV_VERSION} from ${UCSLAM_LIB_OPENCV_DIR})")

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED HINTS "${PANGOLIN_ROOT}/lib/cmake/Pangolin")

if(NOT EXISTS "${UCSLAM_ROOT}/lib/libORB_SLAM3.so")
  message(FATAL_ERROR "UC-SLAM library not found at ${UCSLAM_ROOT}/lib. Run ./build.sh at the repository root first.")
endif()

add_executable(slam_node src/slam_node.cc)
target_include_directories(slam_node PRIVATE
  ${UCSLAM_REPO}/ros/common
  ${UCSLAM_ROOT}
  ${UCSLAM_ROOT}/include
  ${UCSLAM_ROOT}/include/CameraModels
  ${UCSLAM_ROOT}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS})
ament_target_dependencies(slam_node rclcpp std_msgs std_srvs sensor_msgs geometry_msgs nav_msgs diagnostic_msgs
  tf2_ros message_filters ucslam_msgs)
target_link_libraries(slam_node
  ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
  ${UCSLAM_ROOT}/lib/libORB_SLAM3.so
  ${UCSLAM_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${UCSLAM_ROOT}/Thirdparty/g2o/lib/libg2o.so
  boost_serialization crypto)
# The library is not installed: keep its absolute path in the RUNPATH of the node.
set_target_properties(slam_node PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS slam_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
# Every script, so a new one needs no edit here. Rerun cmake (colcon build) after adding one.
file(GLOB UCSLAM_ROS_SCRIPTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py")
install(PROGRAMS ${UCSLAM_ROS_SCRIPTS} DESTINATION lib/${PROJECT_NAME})

ament_package()
