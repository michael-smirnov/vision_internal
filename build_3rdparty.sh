#!/bin/bash

if [ -z "$1" ]; then
    Configuration=Release
else
    Configuration=$1
    if [ $Configuration != "Debug" ] && [ $Configuration != "Release" ]; then
         echo "Configuration must be Debug or Release"
         exit 1
    fi
fi
echo "build configuration is $Configuration"

cd 3rdparty/opencv
if [ -d build ]; then
    echo "build directory exists"
else
    echo "tries to create build directory..."
    mkdir build
    if [ $? == 0 ]; then
        echo "ok"
    fi
fi

cd build

cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_opencv_world=True -D CMAKE_INSTALL_PREFIX="../../../3rdparty-build-$Configuration" .. -G "Visual Studio 15 2017 Win64"
if [ $? != 0 ]; then
    exit 1
fi

MSBuild.exe ALL_BUILD.vcxproj //p:Configuration=$Configuration //m
if [ $? != 0 ]; then
    exit 1
fi

MSBuild.exe INSTALL.vcxproj //p:Configuration=$Configuration //m
if [ $? != 0 ]; then
    exit 1
fi

echo "OpenCV build done"