import qbs

Project {
    name: "vision-internal"

    references: [
        "BackgroundRecognition/BackgroundRecognition.qbs",
        "Autofocus/Autofocus.qbs",
        "Countours/Countours.qbs",
        "Core/Core.qbs"
    ]
}
