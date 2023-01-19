### LIBRARIES ####
add_library(SLucAM_library SHARED
SLucAM_dataset.cpp
SLucAM_geometry.cpp
SLucAM_image.cpp
SLucAM_initialization.cpp
SLucAM_keypoint.cpp
SLucAM_keyframe.cpp
SLucAM_matcher.cpp
SLucAM_measurement.cpp
SLucAM_state.cpp
SLucAM_visualization.cpp
${ANMS_INCLUDE_PATH}/nanoflann.hpp
${ANMS_INCLUDE_PATH}/range-tree/lrtypes.h
${ANMS_INCLUDE_PATH}/range-tree/ranget.h
${ANMS_INCLUDE_PATH}/anms.h)


### EXECUTABLE ###
add_executable(SLucAM main.cpp)


### LINKING ###
target_link_libraries(SLucAM
SLucAM_library
${OpenCV_LIBS}
${EIGEN3_LIBS}
${G2O_LIBS})