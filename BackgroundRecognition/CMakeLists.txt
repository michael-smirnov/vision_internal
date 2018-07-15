cmake_minimum_required (VERSION 2.6)

include_directories ("${PROJECT_SOURCE_DIR}/Core")
include_directories ("${PROJECT_SOURCE_DIR}/3rdparty-build-Debug/include")

add_executable (BackgroundRecognition main.cpp)

set_property (TARGET BackgroundRecognition PROPERTY CXX_STANDARD 17)
target_link_libraries (BackgroundRecognition Core)
target_link_libraries (BackgroundRecognition debug ${OpenCVWorld-Debug}
                                             optimized ${OpenCVWorld-Release})