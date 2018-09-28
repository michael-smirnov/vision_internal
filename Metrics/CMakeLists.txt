cmake_minimum_required (VERSION 2.6)

include_directories ("${PROJECT_SOURCE_DIR}/Core")
include_directories ("${PROJECT_SOURCE_DIR}/3rdparty-build-Debug/include")

add_executable (Metrics main.cpp)

set_property (TARGET Metrics PROPERTY CXX_STANDARD 17)
target_link_libraries (Metrics Core)
target_link_libraries (Metrics debug ${OpenCVWorld-Debug}
                               optimized ${OpenCVWorld-Release})