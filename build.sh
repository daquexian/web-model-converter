#!/usr/bin/env bash

set -xe

pushd $1
source emsdk_env.sh
popd

pushd ./third_party/protobuf
mkdir -p build

pushd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -GNinja ../cmake
ninja
cmake -DCMAKE_INSTALL_PREFIX=install-with-pthreads -P cmake_install.cmake
cmake -DCMAKE_INSTALL_PREFIX=install-without-pthreads -P cmake_install.cmake
popd

mkdir -p build-wasm-without-pthreads
pushd build-wasm-without-pthreads
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Dprotobuf_BUILD_PROTOC_BINARIES=OFF -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -GNinja ../cmake
ninja
cp libprotobuf.a libprotobuf-lite.a ../build/install-without-pthreads/lib/
popd

mkdir -p build-wasm-with-pthreads
pushd build-wasm-with-pthreads
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -Dprotobuf_BUILD_PROTOC_BINARIES=OFF -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -GNinja -DCMAKE_CXX_FLAGS="-pthread -matomics" -DCMAKE_EXE_LINKER_FLAGS="-pthread -matomics" ../cmake
ninja
cp libprotobuf.a libprotobuf-lite.a ../build/install-with-pthreads/lib/
popd

popd

PROTOBUF_WITH_PTHREADS=/home/dev/files/repos/web-model-converter/third_party/protobuf/build/install-with-pthreads/
PROTOBUF_WITHOUT_PTHREADS=/home/dev/files/repos/web-model-converter/third_party/protobuf/build/install-without-pthreads/

pushd third_party/ncnn
git pull
popd

pushd ncnn_wrapper
mkdir -p build
pushd build
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNCNN_SSE2=OFF -DNCNN_BUILD_TOOLS=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITHOUT_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITHOUT_PTHREADS -DLLVM_PROJECT_INSTALL_DIR=/home/dev/files/repos/llvm-project/build-wasm-f4d02fbe/install -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja caffe2ncnn
ninja mxnet2ncnn
ninja onnx2ncnn
ninja darknet2ncnn
ninja ncnnoptimize
ninja mlir2ncnn
popd
popd

pushd third_party/Tengine-Convert-Tools
git pull
popd

pushd tengine_wrapper
mkdir -p build
pushd build
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITH_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITH_PTHREADS -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja tm_convert_tool
popd
popd

pushd onnxopt/onnx-optimizer
git pull
popd

pushd onnxopt
mkdir -p build
pushd build
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITHOUT_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITHOUT_PTHREADS -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja export_onnxopt
popd
popd

pushd build9
emmake make export -j50
popd
