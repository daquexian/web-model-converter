cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

option(protobuf_BUILD_TESTS "Build tests" OFF)
option(protobuf_BUILD_PROTOC_BINARIES "" OFF)
add_subdirectory(third_party/protobuf/cmake)

set(ONNX_CUSTOM_PROTOC_EXECUTABLE /usr/bin/protoc)
add_subdirectory(third_party/onnx)

add_library(onnx2ncnn
    ./onnx2ncnn.cpp
    ./onnx2ncnn.h)
target_link_libraries(onnx2ncnn
    PUBLIC
    onnx)

add_executable(export
    ./export.cpp)
target_link_libraries(export
    PRIVATE
    onnx2ncnn)
target_compile_options(export PRIVATE "SHELL:-s EXPORTED_FUNCTIONS='[_list]'" "SHELL:-s EXTRA_EXPORTED_RUNTIME_METHODS='[ccall,cwrap]'")
