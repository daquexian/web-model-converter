project(paddle_wrapper CXX)
cmake_minimum_required(VERSION 3.10)

option(WITH_LITE "" ON)
option(LITE_ON_MODEL_OPTIMIZE_TOOL "" ON)
option(WITH_TESTING "" OFF)
option(LITE_BUILD_EXTRA "" ON)
option(WITH_MKL "" OFF)
option(LITE_WITH_X86 "" OFF)
option(LITE_WITH_OPENMP "" OFF)
# Please pass PROTOBUF_PROTOC_EXECUTABLE via cli
add_subdirectory(Paddle-Lite)

set_target_properties(opt PROPERTIES LINK_FLAGS "-s EXIT_RUNTIME=1 -s FORCE_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 -s 'EXPORT_NAME=\"create_paddle_opt\"' -s 'EXPORTED_RUNTIME_METHODS=[FS,ccall,cwrap,callMain]' -s EXPORTED_FUNCTIONS=[_main]")
