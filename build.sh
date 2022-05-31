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

# pushd third_party/MNN
# git pull --recurse-submodules
# # NOTE: only do it at first time
# # ./schema/generate.sh
# popd
#
# pushd mnn_wrapper
# mkdir -p build
# pushd build
# emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DMNN_USE_THREAD_POOL=OFF -DMNN_FORBID_MULTI_THREAD=ON -DMNN_BUILD_CONVERTER=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITHOUT_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITHOUT_PTHREADS -GNinja -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_SHARED_LIBS=OFF ..
# # correct dependency is missing in mnn cmake
# # so we need to build MNNCompress in advance of MNNConvert by ourselves
# ninja MNNCompress
# ninja MNNConvert
# popd
# popd

# pushd third_party/ncnn
# git pull --recurse-submodules origin master
# popd
#
# LLVM_SOURCE_DIR=~/files/repos/llvm-project/
# pushd $LLVM_SOURCE_DIR
# LLVM_COMMIT_ID=74e603
# git fetch origin
# git co $LLVM_COMMIT_ID
# HOST_BUILD_DIR=$LLVM_SOURCE_DIR/build-host-$LLVM_COMMIT_ID
# mkdir -p $HOST_BUILD_DIR
# pushd $HOST_BUILD_DIR
# cmake -GNinja -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="" -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_TESTS=OFF ../llvm/
# ninja
# popd
#
# WASM_BUILD_DIR=$LLVM_SOURCE_DIR/build-wasm-$LLVM_COMMIT_ID
# mkdir -p $WASM_BUILD_DIR
# pushd $WASM_BUILD_DIR
# emcmake cmake -GNinja -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="" -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_TESTS=OFF -DMLIR_LINALG_ODS_GEN=$HOST_BUILD_DIR/bin/mlir-linalg-ods-gen -DLLVM_TABLEGEN=$HOST_BUILD_DIR/bin/llvm-tblgen -DMLIR_TABLEGEN=$HOST_BUILD_DIR/bin/mlir-tblgen -DMLIR_LINALG_ODS_YAML_GEN=$HOST_BUILD_DIR/bin/mlir-linalg-ods-yaml-gen ../llvm/
# ninja
# ninja install
# popd
#
# popd

# pushd ncnn_wrapper
# mkdir -p build
# pushd build
# emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DNCNN_SSE2=OFF -DNCNN_BUILD_TOOLS=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITHOUT_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITHOUT_PTHREADS -DLLVM_PROJECT_INSTALL_DIR=$WASM_BUILD_DIR/install -GNinja -DCMAKE_BUILD_TYPE=Release ..
# ninja caffe2ncnn
# ninja mxnet2ncnn
# ninja onnx2ncnn
# ninja darknet2ncnn
# ninja ncnnoptimize
# ninja mlir2ncnn
# popd
# popd
#
# pushd third_party/Tengine-Convert-Tools
# git pull --recurse-submodules origin master
# popd
#
# pushd tengine_wrapper
# mkdir -p build
# pushd build
# emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITH_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITH_PTHREADS -GNinja -DCMAKE_BUILD_TYPE=Release ..
# ninja convert_tool
# popd
# popd

pushd onnxopt/onnx-optimizer
git pull --recurse-submodules origin master
popd

pushd onnxopt
mkdir -p build
pushd build
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_FIND_ROOT_PATH=$PROTOBUF_WITHOUT_PTHREADS -DCMAKE_PREFIX_PATH=$PROTOBUF_WITHOUT_PTHREADS -GNinja -DCMAKE_BUILD_TYPE=Release ..
ninja export_onnxopt
popd
popd

pushd paddle_wrapper

# pushd Paddle-Lite
# git pull --recurse-submodules origin master
# popd

pushd Paddle-Lite/third-party/protobuf-host
git apply ../../cmake/protobuf-host-patch || true
mkdir -p build-protoc

pushd build-protoc
PROTOC_BUILD_DIR=`pwd`
cmake -Dprotobuf_BUILD_TESTS=OFF -GNinja ../cmake
ninja protoc
popd

popd

mkdir -p build
pushd build
emcmake cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DPROTOBUF_PROTOC_EXECUTABLE=$PROTOC_BUILD_DIR/protoc ..
emmake make opt -j`nproc`
popd

popd


# pushd build9
# emmake make export -j50
# popd
