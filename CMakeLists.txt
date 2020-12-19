cmake_minimum_required(VERSION 3.10)

project(wmc CXX)

set(WMC_DIR ${PROJECT_SOURCE_DIR})
# Please use latest emsdk in the case of incompatibility between cmake 3.15 and emsdk
set(CMAKE_CXX_STANDARD 11)

add_compile_options(-Oz)

# find_program(WMC_PROTOC protoc)
message(STATUS "Use protoc at ${WMC_PROTOC}")
set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${WMC_PROTOC})

option(onnxruntime_DISABLE_CONTRIB_OPS "Disable contrib ops" ON)
# TODO: use protobuf-lite
option(onnxruntime_USE_FULL_PROTOBUF "" ON)
add_subdirectory(third_party/onnxruntime/cmake)

# option(protobuf_BUILD_TESTS "Build tests" OFF)
# option(protobuf_BUILD_PROTOC_BINARIES "" OFF)
# add_subdirectory(third_party/protobuf/cmake)

# add_subdirectory(third_party/onnx)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(./third_party/ncnn/tools/mlir)

add_library(mxnet2ncnn
    ./third_party/ncnn/tools/mxnet/mxnet2ncnn.cpp
    ./third_party/ncnn/tools/mxnet/mxnet2ncnn.h)

add_library(onnx2ncnn
    ./third_party/ncnn/tools/onnx/onnx2ncnn.cpp
    ./third_party/ncnn/tools/onnx/onnx2ncnn.h)
target_link_libraries(onnx2ncnn
    PUBLIC
    onnx)
target_include_directories(onnx2ncnn
    PUBLIC
    ./third_party/ncnn/tools/onnx)

include(cmake/protobuf.cmake)
protobuf_generate_cpp(caffe_proto_ncnn_SRCS caffe_proto_ncnn_HDRS third_party/ncnn/tools/caffe/caffe_ncnn.proto)
file(GLOB tf_protos tf_proto/*)
protobuf_generate_cpp(tf_proto_srcs tf_proto_hdrs ${tf_protos})
add_library(singleton_tf_proto ${tf_proto_srcs} ${tf_proto_hdrs})
target_include_directories(singleton_tf_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(singleton_tf_proto protobuf::libprotobuf)

add_library(caffe2ncnn
    ./third_party/ncnn/tools/caffe/caffe2ncnn.cpp
    ./third_party/ncnn/tools/caffe/caffe2ncnn.h
    ${caffe_proto_ncnn_SRCS}
    )
target_link_libraries(caffe2ncnn
    PUBLIC
    protobuf::libprotobuf)
target_include_directories(caffe2ncnn PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        PUBLIC ./third_party/ncnn/tools/caffe/)

option(NCNN_OPENMP "openmp support" OFF)
option(NCNN_BUILD_TESTS "build tests" OFF)
option(NCNN_BUILD_BENCHMARK "build benchmark" OFF)
add_subdirectory(third_party/ncnn)

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
endfunction(add_include)
set_property(GLOBAL PROPERTY proto_list)
function(add_proto)
    get_property(tmp GLOBAL PROPERTY proto_list)
    foreach(arg ${ARGV})
        list(APPEND tmp ${arg})
    endforeach()
    set_property(GLOBAL PROPERTY proto_list "${tmp}")
endfunction(add_proto)

add_source("export.cpp")
# Enable tf model optimizer in MNN converter
option(TFMODEL_OPTIMIZE "Enable tensorflow model optimizer" ON)
option(MNN_BUILD_CONVERTER "" ON)
add_include(third_party/MNN/schema/current)

function(include_directories)
    _include_directories(${ARGV})
    add_include(${ARGV})
endfunction()

add_subdirectory(third_party/MNN)

add_include(third_party/MNN/tools/converter/)
add_include(third_party/MNN/tools/converter/source)
add_include(third_party/MNN/tools/converter/source/include)
add_include(third_party/MNN/tools/converter/source/IR)
add_include(third_party/MNN/tools/converter/source/tflite/schema)
add_include(third_party)

add_subdirectory(third_party/TNN/tools/onnx2tnn/onnx-converter)

include_directories(third_party/TNN/tools/onnx2tnn/src/core)
include_directories(third_party/TNN/tools/onnx2tnn/src/core/objseri)
include_directories(third_party/TNN/tools/onnx2tnn/src/core/half)
include_directories(third_party/TNN/tools/onnx2tnn/src/core/layer)
include_directories(third_party/TNN/tools/onnx2tnn/src/core/encrypt)

file(GLOB_RECURSE ONNX2TNN_SRC
        third_party/TNN/tools/onnx2tnn/src/core/*.h
        third_party/TNN/tools/onnx2tnn/src/core/*.hpp
        third_party/TNN/tools/onnx2tnn/src/core/*.cc
        third_party/TNN/tools/onnx2tnn/src/core/*.cpp)

add_library(onnx2tnn ${ONNX2TNN_SRC} ${WMC_PROTOC})

target_link_libraries(onnx2tnn PUBLIC onnx)


option(CONFIG_BUILD_SERIALIZER "" ON)
add_subdirectory(third_party/tengine/)

get_property(export_srcs GLOBAL PROPERTY source_list)
get_property(include_dirs GLOBAL PROPERTY include_dirs_list)
get_property(protos GLOBAL PROPERTY proto_list)
protobuf_generate_cpp(proto_srcs proto_hdrs
    ${protos})

add_definitions(-DMNN_USE_THREAD_POOL)
add_executable(export
    ${export_srcs}
    ${proto_srcs}
    )
target_link_libraries(export
    PRIVATE
    onnx2ncnn
    caffe2ncnn
    mxnet2ncnn
    mlir2ncnn
    tengine-onnx-serializer
    tengine-caffe-serializer
    tengine-tensorflow-serializer
    tengine-mxnet-serializer
    tengine-darknet-serializer
    tengine-tflite-serializer
    tengine
    ncnnoptimize
    onnx
    singleton_tf_proto
    onnx_test_runner
    onnx2tnn
    # MNNConverterCaffe
    # mnn_bizcode tflite optimizer COMMON_LIB
    # MNNConvertDeps
    )
target_include_directories(export
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${include_dirs}
    third_party/tengine/core/include
    third_party/tengine/include
    )
set_target_properties(export PROPERTIES LINK_FLAGS "-s DISABLE_EXCEPTION_CATCHING=0 -s FILESYSTEM=0 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS=[_onnx2tnn_export,_mlir2ncnn_export,_mxnet2ncnn_export,_caffe2ncnn_export,_onnx2ncnn_export,_caffe2mnn_export,_onnx2mnn_export,_tf2mnn_export,_onnx2tengine_export,_caffe2tengine_export,_tf2tengine_export,_mxnet2tengine_export,_darknet2tengine_export,_tflite2tengine_export,_ncnnoptimize_export,_onnxoptimize_export,_check_static_input_size_export,_onnxsimplify_export,_onnx_shape_infer_export,_create_exporter,_free_exporter,_get_buffer1,_get_buffer2,_get_buffer_size1,_get_buffer_size2,_get_buffer3,_get_buffer_size3] -s EXTRA_EXPORTED_RUNTIME_METHODS=[ccall,cwrap]")

add_subdirectory(onnxopt)
