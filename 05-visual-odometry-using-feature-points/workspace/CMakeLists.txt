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
# g2o:
SET(SUITESPARSE_INCLUDE_DIR "/usr/include/suitesparse/")
SET(G2O_INCLUDE_DIR "/usr/local/include/g2o")
# Pangolin:
FIND_PACKAGE(Pangolin REQUIRED)

# header files:
INCLUDE_DIRECTORIES(
    ${EIGEN3_INCLUDE_DIR}
    ${SOPHUS_INCLUDE_DIR}
    ${SUITESPARSE_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
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

# 04: ICP for pose estimation:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/04-ICP)
ADD_EXECUTABLE(trajectory-matching 04-ICP/trajectory-matching.cpp)
TARGET_LINK_LIBRARIES(
    trajectory-matching
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    Sophus
)

# 05: PnP through g2o:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/05-pnp-using-g2o)
ADD_EXECUTABLE(pnp-using-g2o 05-pnp-using-g2o/pnp-using-g2o.cpp)
TARGET_LINK_LIBRARIES(
    pnp-using-g2o
    /usr/lib/x86_64-linux-gnu/libcolamd.so
    /usr/lib/x86_64-linux-gnu/libklu.so
    /usr/lib/x86_64-linux-gnu/libcholmod.so
    /usr/lib/x86_64-linux-gnu/libldl.so
    /usr/lib/x86_64-linux-gnu/libcamd.so
    /usr/lib/x86_64-linux-gnu/libamd.so
    /usr/lib/x86_64-linux-gnu/libcsparse.so
    /usr/lib/x86_64-linux-gnu/libbtf.so
    /usr/lib/x86_64-linux-gnu/libumfpack.so
    /usr/lib/x86_64-linux-gnu/libcxsparse.so
    /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
    /usr/lib/x86_64-linux-gnu/libccolamd.so
    /usr/lib/x86_64-linux-gnu/libspqr.so
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
)

# 05: PnP through g2o:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/06-ICP-using-g2o)
ADD_EXECUTABLE(ICP-using-g2o 06-ICP-using-g2o/ICP-using-g2o.cpp)
TARGET_LINK_LIBRARIES(
    ICP-using-g2o
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    Sophus
    /usr/local/lib/libg2o_core.so 
    /usr/local/lib/libg2o_stuff.so 
    /usr/local/lib/libg2o_types_sba.so 
    /usr/local/lib/libg2o_csparse_extension.so
)