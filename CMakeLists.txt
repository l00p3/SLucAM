# CMake version
cmake_minimum_required(VERSION 3.12)

# Project name
project(SLucAM VERSION 1.0)

# C++ version
set(CMAKE_CXX_STANDARD 17)

### OPENCV ###
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

### Eigen ###
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

### g2o ###
add_definitions(-DG2O_USE_VENDORED_CERES)
list(APPEND G2O_LIBS
    ${PROJECT_SOURCE_DIR}/external/g2o/lib/libg2o_stuff.so
    ${PROJECT_SOURCE_DIR}/external/g2o/lib/libg2o_core.so
    ${PROJECT_SOURCE_DIR}/external/g2o/lib/libg2o_types_sba.so
    ${PROJECT_SOURCE_DIR}/external/g2o/lib/libg2o_solver_dense.so)
include_directories(${PROJECT_SOURCE_DIR}/external/g2o)

### ANMS ###
include_directories(${PROJECT_SOURCE_DIR}/external/ANMS)
set(ANMS_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/external/ANMS)

# Where to put binary and library files
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Where to find the headers to include with #include <...>
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the CMakeLists.txt of the src folder
add_subdirectory(src)