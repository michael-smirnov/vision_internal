cmake_minimum_required (VERSION 2.6)

include_directories ("${PROJECT_SOURCE_DIR}/3rdparty-build-Debug/include")

add_library (Core   AverageCalculator.cpp 
                    ComponentMarker.cpp 
                    TraceFinder.cpp
                    ImageProcessor.cpp
                    MathItemGenerator.cpp
                    Statistics.cpp
                    Utility.cpp)

set_property (TARGET Core PROPERTY CXX_STANDARD 17)
target_link_libraries (Core debug ${OpenCVWorld-Debug}
                            optimized ${OpenCVWorld-Release})