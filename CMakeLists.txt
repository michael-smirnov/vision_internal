cmake_minimum_required (VERSION 2.6)

project (VisionInternal)

find_library (OpenCVWorld-Debug opencv_world400d PATHS 3rdparty-build-Debug/x64/vc15/lib)
find_library (OpenCVWorld-Release opencv_world400 PATHS 3rdparty-build-Release/x64/vc15/lib)

add_subdirectory (Core)
add_subdirectory (Countours)