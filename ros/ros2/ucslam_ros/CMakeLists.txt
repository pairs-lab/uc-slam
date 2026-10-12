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
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(ucslam_msgs REQUIRED)
find_package(OpenCV 4 REQUIRED)
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
  tf2_ros cv_bridge message_filters ucslam_msgs)
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
install(PROGRAMS scripts/asl_to_bag.py DESTINATION lib/${PROJECT_NAME})

ament_package()
