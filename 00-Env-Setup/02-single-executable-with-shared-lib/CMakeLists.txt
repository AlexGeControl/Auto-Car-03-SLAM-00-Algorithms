# minimum cmake version:
cmake_minimum_required( VERSION 3.5 )

# project name:
project( HelloVisualSLAM )

# libraries:
add_library( interface SHARED lib-hello-slam.cpp )

# executables:
add_executable( main main.cpp )
target_link_libraries( main interface )
