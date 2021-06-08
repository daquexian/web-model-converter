project(tengine_wrapper CXX)
cmake_minimum_required(VERSION 3.10)

set(WMC_DIR ${PROJECT_SOURCE_DIR}/..)

add_subdirectory(${WMC_DIR}/third_party/Tengine-Convert-Tools ${CMAKE_CURRENT_BINARY_DIR}/tengine)

set_target_properties(convert_tool PROPERTIES LINK_FLAGS "-s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=128MB -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_x2tengine\"' -s 'EXTRA_EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
