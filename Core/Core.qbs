import qbs

StaticLibrary {
    name: "Core"

    Depends { name: "cpp" }

    files: [
        "AverageCalculator.h",
        "functions.h",
        "AverageCalculator.cpp",
        "functions.cpp",
    ]

    cpp.includePaths: [
        "/usr/local/include/"
    ]

    cpp.libraryPaths: [
        "/usr/local/lib/"
    ]

    cpp.staticLibraries: [
        "opencv_world"
    ]

    cpp.cxxLanguageVersion: "c++17"

}
