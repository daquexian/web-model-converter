#!/usr/bin/env bash

set -xe

pushd $1
source emsdk_env.sh
popd

PROTOBUF_WITH_PTHREADS=/home/dev/files/repos/web-model-converter/third_party/protobuf/cmake/build/install/
PROTOBUF_WITHOUT_PTHREADS=/home/dev/files/repos/web-model-converter/third_party/protobuf/cmake/build/install-no-pthread

pushd third_party/ncnn
git pull
popd

pushd ncnn_wrapper
mkdir -p build
pushd build
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNCNN_SSE2=OFF -DNCNN_BUILD_TOOLS=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITHOUT_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITHOUT_PTHREADS -DLLVM_PROJECT_INSTALL_DIR=/home/dev/files/repos/llvm-project/build-wasm-f4d02fbe/install -GNinja ..
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
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITH_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITH_PTHREADS -GNinja ..
ninja tm_convert_tool
popd
popd

pushd onnxopt/onnx-optimizer
git pull
popd

pushd onnxopt
mkdir -p build
pushd build
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITHOUT_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITHOUT_PTHREADS -GNinja ..
ninja export_onnxopt
popd
popd

pushd build9
emmake make export -j50
popd
