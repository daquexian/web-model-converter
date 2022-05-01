cmake_minimum_required(VERSION 3.10)

project(wmc_onnx_opt CXX)

add_compile_options(-Oz)

set(TOP_DIR ${PROJECT_SOURCE_DIR}/..)
add_subdirectory(onnx-optimizer)

add_executable(export_onnxopt
    export_onnxopt.cpp)
target_link_libraries(export_onnxopt
    onnx_optimizer)
target_include_directories(export_onnxopt
    PRIVATE
    ${TOP_DIR}/common)
set_target_properties(export_onnxopt PROPERTIES LINK_FLAGS "-s DISABLE_EXCEPTION_CATCHING=0 -s FILESYSTEM=0 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"createOnnxOpt\"' -s EXPORTED_FUNCTIONS=[_onnxoptimize_export,_onnx_shape_infer_export,_create_exporter,_free_exporter,_get_buffer1,_get_buffer2,_get_buffer_size1,_get_buffer_size2,_get_buffer3,_get_buffer_size3] -s EXPORTED_RUNTIME_METHODS=[ccall,cwrap]")
