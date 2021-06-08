project(mnn_wrapper CXX)
cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0079 NEW)

set(WMC_DIR ${PROJECT_SOURCE_DIR}/..)

add_subdirectory(${WMC_DIR}/third_party/MNN ${CMAKE_CURRENT_BINARY_DIR}/mnn)

set_target_properties(MNNConvert PROPERTIES LINK_FLAGS "-s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=128MB -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_x2mnn\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")

