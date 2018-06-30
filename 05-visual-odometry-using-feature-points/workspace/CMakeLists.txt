CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(OBSERVATION_MODEL)

# OpenCV3:
FIND_PACKAGE(OpenCV 3.0 QUIET)
IF(NOT OpenCV_FOUND)
   FIND_PACKAGE(OpenCV 2.4.3 QUIET)
   IF(NOT OpenCV_FOUND)
      MESSAGE(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   ENDIF()
ENDIF()
# Eigen3:
FIND_PACKAGE(Eigen3 REQUIRED)
# Sophus:
SET(SOPHUS_INCLUDE_DIR "/home/alexge/svslocal/include")

# header files:
INCLUDE_DIRECTORIES(
    ${EIGEN3_INCLUDE_DIR}
    ${SOPHUS_INCLUDE_DIR}
)
# shared libraries:
LINK_DIRECTORIES(/home/alexge/svslocal/lib)

# runnable:
ADD_COMPILE_OPTIONS(-std=c++11)

# 01: Image Undistortion:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/01-ORB)
ADD_EXECUTABLE(compute-orb 01-ORB/computeORB.cpp)
TARGET_LINK_LIBRARIES(
    compute-orb
    ${OpenCV_LIBS}
)

# 02. Pose Estimation from Essential Matrix:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/02-camera-pose-estimation)
ADD_EXECUTABLE(pose-estimation 02-camera-pose-estimation/pose-estimation.cpp)
TARGET_LINK_LIBRARIES(
    pose-estimation
    ${EIGEN3_LIBS}
    Sophus
)

# 03: Bundle Adjustment through Gaussian Newton:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/03-pnp-using-bundle-adjustment)
ADD_EXECUTABLE(GN-BA 03-pnp-using-bundle-adjustment/GN-BA.cpp)
TARGET_LINK_LIBRARIES(
    GN-BA
    ${EIGEN3_LIBS}
    Sophus
)