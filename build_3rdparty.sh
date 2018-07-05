#!/bin/bash

function FindBuildSystem
{
    specificBuildSystem=$1
    which $specificBuildSystem
    if [ $? == 0 ]; then
        BuildSystem=$specificBuildSystem
        return 0
    fi

    which MSBuild.exe
    if [ $? == 0 ]; then
        BuildSystem="MSBuild.exe"
        return 0
    fi

    which make
    if [ $? == 0 ]; then 
        BuildSystem="make"
        return 0
    fi

    which mingw32-make
    if [ $? == 0 ]; then 
        BuildSystem="mingw32-make"
        return 0
    fi
    
    return 1
}

function InstallBuildVariables
{
    ProcessesCount="$(grep -c ^processor /proc/cpuinfo)"

    if [ $BuildSystem == "MSBuild.exe" ]; then
        CmakeAdditionalFlags='-G "Visual Studio 15 2017 Win64"'
        BuildCmd="$BuildSystem ALL_BUILD.vcxproj //p:Configuration=$Configuration //m"
        InstallCmd="$BuildSystem INSTALL.vcxproj //p:Configuration=$Configuration //m"
    elif [ $BuildSystem == "mingw32-make" ]; then
        CmakeAdditionalFlags='-D CMAKE_SH="CMAKE_SH-NOTFOUND" -D ENABLE_PRECOMPILED_HEADERS=OFF -D WITH_IPP=OFF -D WITH_TBB=OFF -D WITH_MSMF=OFF -G "MinGW Makefiles"'
        BuildCmd="$BuildSystem -j $ProcessesCount"
        InstallCmd="$BuildSystem install"
    elif [ $BuildSystem == "make" ]; then
        CmakeAdditionalFlags='-D ENABLE_PRECOMPILED_HEADERS=OFF -D WITH_IPP=OFF -D WITH_TBB=OFF -D WITH_MSMF=OFF -G "MinGW Makefiles"'
        BuildCmd="$BuildSystem -j $ProcessesCount"
        InstallCmd="$BuildSystem install"
    else
        return 1
    fi

    return 0
}

if [[ -z "$1" || ( "$1" != "clean" && "$1" != "build" ) ]]; then
    echo "build_3rdparty build|clean [Debug|Release]"
    exit 1
elif [ "$1" == "clean" ]; then
    cd 3rdparty/opencv
    rm -rf build
    echo "Clean done"
    exit 0
fi

if [ -z "$2" ]; then
    Configuration=Release
else
    Configuration=$2
    if [ $Configuration != "Debug" ] && [ $Configuration != "Release" ]; then
         echo "Configuration must be Debug or Release"
         exit 1
    fi
fi
echo "build configuration is $Configuration"

echo "Check cmake ..."
which cmake &> /dev/null
if [ $? != 0 ]; then
    echo "Check cmake ... FAIL"
    exit 1
else
    echo "Check cmake ... OK"
fi

echo "Check build system ..."
FindBuildSystem &> /dev/null
if [ $? != 0 ]; then
    echo "Check build system ... Fail"
    exit 1
else
    echo "Check build system ... $BuildSystem"
fi

InstallBuildVariables
if [ $? != 0 ]; then
    echo "Can not install build variables"
    exit 1
fi

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

eval cmake -D CMAKE_BUILD_TYPE=$Configuration -D BUILD_opencv_world=True -D CMAKE_INSTALL_PREFIX="../../../3rdparty-build-$Configuration" "$CmakeAdditionalFlags" ..
if [ $? != 0 ]; then
    exit 1
fi

eval $BuildCmd
if [ $? != 0 ]; then
    exit 1
fi

eval $InstallCmd
if [ $? != 0 ]; then
    exit 1
fi

echo "OpenCV build done"