cmake_minimum_required(VERSION 3.10)
project(ucslam_ros1)

# UC-SLAM is built first by ./build.sh at the repository root. This package links its library. The package is often
# a symlink in a catkin workspace, so the repository is found from its real path.
get_filename_component(UCSLAM_PKG_REAL "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(UCSLAM_REPO_DEFAULT "${UCSLAM_PKG_REAL}/../../.." ABSOLUTE)
set(UCSLAM_REPO "${UCSLAM_REPO_DEFAULT}" CACHE PATH "UC-SLAM repository root")
set(UCSLAM_ROOT "${UCSLAM_REPO}/cpp/ucslam" CACHE PATH "UC-SLAM source and build tree")
set(PANGOLIN_ROOT "${UCSLAM_REPO}/third_party/Pangolin/install" CACHE PATH "Pangolin install of ./build.sh")

# C++14 for Noetic (the library uses C++11, the same ABI), with the same compiler and -march as the library, so
# that Eigen objects have the same layout on both sides.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

find_package(catkin REQUIRED COMPONENTS
  roscpp rospy std_msgs std_srvs sensor_msgs geometry_msgs nav_msgs diagnostic_msgs tf2_ros cv_bridge
  message_filters message_generation)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED HINTS "${PANGOLIN_ROOT}/lib/cmake/Pangolin")

add_message_files(FILES DepthRequest.msg DepthPrior.msg)
generate_messages(DEPENDENCIES std_msgs sensor_msgs)

catkin_package(CATKIN_DEPENDS message_runtime roscpp std_msgs sensor_msgs)

if(NOT EXISTS "${UCSLAM_ROOT}/lib/libORB_SLAM3.so")
  message(FATAL_ERROR "UC-SLAM library not found at ${UCSLAM_ROOT}/lib. Run ./build.sh at the repository root first.")
endif()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${UCSLAM_REPO}/ros/common
  ${UCSLAM_ROOT}
  ${UCSLAM_ROOT}/include
  ${UCSLAM_ROOT}/include/CameraModels
  ${UCSLAM_ROOT}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS})

add_executable(slam_node src/slam_node.cc)
add_dependencies(slam_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(slam_node
  ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
  ${UCSLAM_ROOT}/lib/libORB_SLAM3.so
  ${UCSLAM_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${UCSLAM_ROOT}/Thirdparty/g2o/lib/libg2o.so
  -lboost_serialization -lcrypto)

catkin_install_python(PROGRAMS scripts/depth_node.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(TARGETS slam_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
# The launch file starts depth_node through $(find ucslam_ros1)/scripts/venv_python.sh. In an install space it needs
# UCSLAM_REPO set to the repository.
install(PROGRAMS scripts/venv_python.sh DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts)
set_target_properties(slam_node PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
