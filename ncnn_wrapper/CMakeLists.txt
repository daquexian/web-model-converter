project(ncnn_wrapper CXX)
cmake_minimum_required(VERSION 3.10)

set(WMC_DIR ${PROJECT_SOURCE_DIR}/..)

add_subdirectory(${WMC_DIR}/third_party/ncnn ${CMAKE_CURRENT_BINARY_DIR}/ncnn)

set_target_properties(caffe2ncnn PROPERTIES LINK_FLAGS "-s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_caffe2ncnn\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
set_target_properties(mxnet2ncnn PROPERTIES LINK_FLAGS "-s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_mxnet2ncnn\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
set_target_properties(onnx2ncnn PROPERTIES LINK_FLAGS "-s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_onnx2ncnn\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
set_target_properties(darknet2ncnn PROPERTIES LINK_FLAGS "-s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_darknet2ncnn\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
set_target_properties(ncnnoptimize PROPERTIES LINK_FLAGS "-s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_ncnnoptimize\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")

set(LLVM_PROJECT_INSTALL_DIR ${LLVM_PROJECT_INSTALL_DIR} CACHE STRING "")
add_subdirectory(${WMC_DIR}/third_party/ncnn/tools/mlir ${CMAKE_CURRENT_BINARY_DIR}/mlir2ncnn)
set_target_properties(mlir2ncnn PROPERTIES LINK_FLAGS "-s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_mlir2ncnn\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
