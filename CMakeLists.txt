cmake_minimum_required(VERSION 3.15)
project(SemanticSegmentation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV CONFIG REQUIRED)

# --- Manual ONNX Runtime setup (official prebuilt, not vcpkg) ---
set(ORT_ROOT "C:/Users/Sukhada/libs/onnxruntime-win-x64-1.25.0")

add_executable(inference src/main.cpp src/segmentation.cpp)

target_include_directories(inference PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_ROOT}/include
)

target_link_libraries(inference PRIVATE
    ${OpenCV_LIBS}
    ${ORT_ROOT}/lib/onnxruntime.lib
)

# Copy the DLL next to the .exe after build — onnxruntime.dll is loaded
# at RUNTIME (dynamic linking), not embedded into inference.exe, so it
# must physically sit beside the .exe or be on PATH, or you'll get a
# "missing DLL" error at launch even though the build succeeds.
add_custom_command(TARGET inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:inference>"
)