cmake_minimum_required(VERSION 3.10)

project(wmc CXX)

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
protobuf_generate_cpp(caffe_proto_ncnn_SRCS caffe_proto_ncnn_HDRS caffe_ncnn.proto)

add_library(caffe2ncnn
	./caffe2ncnn.cpp
	./caffe2ncnn.h
    ${caffe_proto_ncnn_SRCS}
    )
target_link_libraries(caffe2ncnn
    PUBLIC
    protobuf::libprotobuf)
target_include_directories(caffe2ncnn PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

set_property(GLOBAL PROPERTY source_list)
function(add_source)
    get_property(tmp GLOBAL PROPERTY source_list)
    foreach(arg ${ARGV})
        list(APPEND tmp ${arg})
    endforeach()
    set_property(GLOBAL PROPERTY source_list "${tmp}")
endfunction(add_source)
set_property(GLOBAL PROPERTY include_dirs_list)
function(add_include)
    get_property(tmp GLOBAL PROPERTY include_dirs_list)
    foreach(arg ${ARGV})
        list(APPEND tmp ${arg})
    endforeach()
    set_property(GLOBAL PROPERTY include_dirs_list "${tmp}")
endfunction(add_source)

add_source("export.cpp")
# set(EXPORT_SRC "export.cpp" CACHE INTERNAL "" FORCE)
add_subdirectory(third_party/MNN/tools/converter/)
add_include(third_party/MNN/tools/converter/)
add_include(third_party/MNN/tools/converter/source/include)
add_include(third_party/MNN/tools/converter/source/IR)
add_include(third_party/MNN/tools/converter/source/tflite/schema)

get_property(export_src GLOBAL PROPERTY source_list)
get_property(include_dirs GLOBAL PROPERTY include_dirs_list)

add_executable(export
    ${export_src})
target_link_libraries(export
    PRIVATE
    onnx2ncnn
    caffe2ncnn
    tensorflow onnx2mnn mnn_bizcode tflite optimizer COMMON_LIB
    )
target_include_directories(export
    PRIVATE
    ${include_dirs})
# target_compile_options(export PRIVATE "SHELL:-s EXPORTED_FUNCTIONS='[_list]'" "SHELL:-s EXTRA_EXPORTED_RUNTIME_METHODS='[ccall,cwrap]'")
set_target_properties(export PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS=[_caffe2ncnn_export,_onnx2ncnn_export,_caffe2mnn_export,_create_exporter,_free_exporter,_get_buffer1,_get_buffer2,_get_buffer_size1,_get_buffer_size2] -s EXTRA_EXPORTED_RUNTIME_METHODS=[ccall,cwrap]")
