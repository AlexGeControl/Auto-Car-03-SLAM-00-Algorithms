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
# Pangolin:
FIND_PACKAGE(Pangolin REQUIRED)

# header files:
INCLUDE_DIRECTORIES(
    ${EIGEN3_INCLUDE_DIR}
    ${SOPHUS_INCLUDE_DIR}
)
# shared libraries:
LINK_DIRECTORIES(/home/alexge/svslocal/lib)

# runnable:
ADD_COMPILE_OPTIONS(-std=c++11)

# 01 - Lukas Kanade:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/01-lukas-kanade)
ADD_EXECUTABLE(optical-flow 01-lukas-kanade/optical-flow.cpp)
TARGET_LINK_LIBRARIES(
    optical-flow
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
)

# 02 - Direct Method:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/02-direct-method)
ADD_EXECUTABLE(direct-method 02-direct-method/direct-method.cpp)
TARGET_LINK_LIBRARIES(
    direct-method
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    Sophus
    ${Pangolin_LIBRARIES}
)