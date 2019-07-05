cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)

option(protobuf_BUILD_TESTS "Build tests" OFF)
option(protobuf_BUILD_PROTOC_BINARIES "" OFF)
add_subdirectory(third_party/protobuf/cmake)

set(WMC_PROTOC /usr/bin/protoc)
set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${WMC_PROTOC})
add_subdirectory(third_party/onnx)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(onnx2ncnn
    ./onnx2ncnn.cpp
    ./onnx2ncnn.h)
target_link_libraries(onnx2ncnn
    PUBLIC
    onnx)

include(cmake/protobuf.cmake)
protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS caffe.proto)
add_library(caffe2ncnn
	./caffe2ncnn.cpp
	./caffe2ncnn.h
    ${CAFFE_PROTO_SRCS}
    ${CAFFE_PROTO_HDRS}
    )
target_include_directories(caffe2ncnn PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(caffe2ncnn
    PUBLIC
    protobuf::libprotobuf)

add_executable(export
    ./export.cpp)
target_link_libraries(export
    PRIVATE
    onnx2ncnn
    caffe2ncnn)
# target_compile_options(export PRIVATE "SHELL:-s EXPORTED_FUNCTIONS='[_list]'" "SHELL:-s EXTRA_EXPORTED_RUNTIME_METHODS='[ccall,cwrap]'")


