cmake_minimum_required (VERSION 2.6)

include_directories ("${PROJECT_SOURCE_DIR}/Core")
include_directories ("${PROJECT_SOURCE_DIR}/3rdparty-build-Debug/include")

add_executable (Countours main.cpp
                          HistogramModaCalculator.cpp
                          SingularPointsFinder.cpp)

set_property (TARGET Countours PROPERTY CXX_STANDARD 17)
target_link_libraries (Countours Core)
target_link_libraries (Countours debug ${OpenCVWorld-Debug}
                                 optimized ${OpenCVWorld-Release})

