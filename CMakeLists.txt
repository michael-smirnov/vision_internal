cmake_minimum_required (VERSION 2.6)

project (VisionInternal)

if (CMAKE_COMPILER_IS_GNUCXX)
    if (WIN32)
        message ("Compiler - mingw")

        set(DEBUG_OPENCV_LIB_NAME "opencv_world400d")
        set(RELEASE_OPENCV_LIB_NAME "opencv_world400")

        set(3RDPARTY_LIBS "x64/mingw/lib")
    elseif(UNIX)
        message ("Compiler - gcc")

        set(DEBUG_OPENCV_LIB_NAME "opencv_world")
        set(RELEASE_OPENCV_LIB_NAME "opencv_world")

        set(3RDPARTY_LIBS "lib")
    else()
        message (FATAL_ERROR "unavaliable OS" )
    endif()
elseif (MSVC)
    if (MSVC_VERSION GREATER_EQUAL 1500)
        message ("Compiler - vc15")

        set(DEBUG_OPENCV_LIB_NAME "opencv_world400d")
        set(RELEASE_OPENCV_LIB_NAME "opencv_world400")

        set (3RDPARTY_LIBS "x64/vc15/lib")
    else()
        message (FATAL_ERROR "unavaliable Visual Studio version")
    endif()
endif()

find_library (OpenCVWorld-Debug ${DEBUG_OPENCV_LIB_NAME} PATHS 3rdparty-build-Debug/${3RDPARTY_LIBS})
find_library (OpenCVWorld-Release ${RELEASE_OPENCV_LIB_NAME} PATHS 3rdparty-build-Release/${3RDPARTY_LIBS})

add_subdirectory (Core)
add_subdirectory (Autofocus)
add_subdirectory (BackgroundRecognition)
add_subdirectory (Countours)
add_subdirectory (Metrics)