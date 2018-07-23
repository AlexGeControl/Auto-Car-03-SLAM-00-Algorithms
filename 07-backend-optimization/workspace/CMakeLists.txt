CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(BACKEND_OPTIMIZATION)

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
# Suitesparse:
SET(SUITESPARSE_INCLUDE_DIR "/usr/include/suitesparse/")
# Ceres:
FIND_PACKAGE(Ceres REQUIRED)
# Pangolin:
FIND_PACKAGE(Pangolin REQUIRED)

# header files:
INCLUDE_DIRECTORIES(
    ${EIGEN3_INCLUDE_DIR}
    ${SOPHUS_INCLUDE_DIR}
    ${SUITESPARSE_INCLUDE_DIR}
)
# shared libraries:
LINK_DIRECTORIES(/home/alexge/svslocal/lib)
SET(
    G2O_LIBRARIES
    /usr/local/lib/libg2o_cli.so                   
    /usr/local/lib/libg2o_interface.so       
    /usr/local/lib/libg2o_solver_dense.so           
    /usr/local/lib/libg2o_types_data.so           
    /usr/local/lib/libg2o_types_slam2d.so
    /usr/local/lib/libg2o_core.so                  
    /usr/local/lib/libg2o_opengl_helper.so   
    /usr/local/lib/libg2o_solver_eigen.so           
    /usr/local/lib/libg2o_types_icp.so            
    /usr/local/lib/libg2o_types_slam3d_addons.so
    /usr/local/lib/libg2o_csparse_extension.so     
    /usr/local/lib/libg2o_parser.so          
    /usr/local/lib/libg2o_solver_pcg.so             
    /usr/local/lib/libg2o_types_sba.so            
    /usr/local/lib/libg2o_types_slam3d.so
    /usr/local/lib/libg2o_ext_freeglut_minimal.so  
    /usr/local/lib/libg2o_simulator.so       
    /usr/local/lib/libg2o_solver_slam2d_linear.so   
    /usr/local/lib/libg2o_types_sclam2d.so        
    /usr/local/lib/libg2o_viewer.so
    /usr/local/lib/libg2o_incremental.so           
    /usr/local/lib/libg2o_solver_cholmod.so  
    /usr/local/lib/libg2o_solver_structure_only.so  
    /usr/local/lib/libg2o_types_sim3.so
    /usr/local/lib/libg2o_interactive.so           
    /usr/local/lib/libg2o_solver_csparse.so  
    /usr/local/lib/libg2o_stuff.so                  
    /usr/local/lib/libg2o_types_slam2d_addons.so
)

INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/01-BAL-g2o/BALDataset
    ${PROJECT_SOURCE_DIR}/01-BAL-g2o/BALTypes
    ${PROJECT_SOURCE_DIR}/01-BAL-g2o/utils
)

ADD_LIBRARY(BALDataset SHARED ${PROJECT_SOURCE_DIR}/01-BAL-g2o/BALDataset/BALDataset.cpp)

# compilation options:
SET(CMAKE_CXX_STANDARD 11)

# 01 - BAL on Dubrovnik:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/01-BAL-g2o)
ADD_EXECUTABLE(bal 01-BAL-g2o/main.cpp)
TARGET_LINK_LIBRARIES(
    bal
    BALDataset
    ${CERES_LIBRARIES}
    ${G2O_LIBRARIES}
)

# 02 - BAL for direct method:
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/02-direct-g2o)
ADD_EXECUTABLE(directBA 02-direct-g2o/directBA.cpp)
TARGET_LINK_LIBRARIES(
    directBA
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    Sophus
    ${G2O_LIBRARIES}
    ${Pangolin_LIBRARIES}
)