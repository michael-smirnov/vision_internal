import qbs

CppApplication {
    name: "Countours"

    Depends { name: "Core" }

    files: [
        "main.cpp",
    ]

    cpp.includePaths: [
        "/usr/local/include/",
        "../Core/",
    ]

    cpp.libraryPaths: [
        "/usr/local/lib/"
    ]

    cpp.staticLibraries: [
        "opencv_world",
    ]

    cpp.cxxLanguageVersion: "c++17"

}
