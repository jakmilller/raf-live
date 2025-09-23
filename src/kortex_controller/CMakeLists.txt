cmake_minimum_required(VERSION 3.8)
project(kortex_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(raf_interfaces REQUIRED)

# Kortex API paths (local to this package)
set(KORTEX_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kortex_api")
set(KORTEX_INCLUDE_DIR "${KORTEX_API_DIR}/include")
set(KORTEX_LIB_RELEASE "${KORTEX_API_DIR}/lib/release/libKortexApiCpp.a")

# Verify Kortex API exists
if(NOT EXISTS ${KORTEX_INCLUDE_DIR})
    message(FATAL_ERROR "Kortex API headers not found at ${KORTEX_INCLUDE_DIR}. Please extract the kortex_api to ${KORTEX_API_DIR}")
endif()

if(NOT EXISTS ${KORTEX_LIB_RELEASE})
    message(FATAL_ERROR "Kortex API library not found at ${KORTEX_LIB_RELEASE}")
endif()

message(STATUS "Using Kortex API from: ${KORTEX_API_DIR}")

# Include directories
include_directories(include)
include_directories(${KORTEX_INCLUDE_DIR})
include_directories(${KORTEX_INCLUDE_DIR}/client)
include_directories(${KORTEX_INCLUDE_DIR}/client_stubs)
include_directories(${KORTEX_INCLUDE_DIR}/common)
include_directories(${KORTEX_INCLUDE_DIR}/messages)

# Create executable
add_executable(controller
  src/controller.cpp
)

# Set ROS2 dependencies
ament_target_dependencies(controller
  rclcpp
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  tf2_ros
  tf2_geometry_msgs
  raf_interfaces
)

# Link Kortex API library
target_link_libraries(controller
  ${KORTEX_LIB_RELEASE}
  pthread
)

# Install executable
install(TARGETS controller
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
)

# Install Python scripts
install(PROGRAMS
  scripts/robot_controller_ros2.py
  scripts/orchestrator.py
  scripts/face_detection.py
  scripts/visual_servoing.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()