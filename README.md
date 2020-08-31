## Compile and install protoc 3.7.0

因为项目用到了 onnxruntime 里的 protobuf，所以需要同样版本的 protoc，我们去 protobuf 源码目录里编译安装

1. `cd third_party/onnxruntime/cmake/external/protobuf/cmake/`
2. `mkdir build && cd build`
3. `cmake ..`
4. `make protoc -j$(nproc)`
5. `cmake -DCMAKE_INSTALL_PREFIX=install -P cmake_install.cmake` 这一步是把编译好的文件安装到 install 目录下

## Generate MNN schema (Only do it once)

1. `cd third_party/MNN`
2. `./schema/generate.sh`

## emsdk environment

1. install emsdk
2. run `source emsdk_env.sh`

## Compile mlir

1. cmake for build host tools: `cmake -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_TARGETS_TO_BUILD=Native -GNinja ../llvm/`
2. Change mlir/include/mlir/Dialect/Linalg/IR/CMakeLists.txt:
"COMMAND mlir-linalg-ods-gen -gen-ods-decl ${TC_SOURCE} > ${GEN_ODS_FILE}"
Replace mlir-linalg-ods-gen to the path of the host tool
2. cmake for build wasm: `emcmake cmake -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_TARGETS_TO_BUILD=Native -DLLVM_TABLEGEN=`pwd`/../build-shared/bin/llvm-tblgen -DMLIR_TABLEGEN=`pwd`/../build-shared/bin/mlir-tblgen ../llvm/`

## Compile convertmodel.com

3. `mkdir build && cd build`
4. run `emcmake cmake -DWMC_PROTOC=$PWD/../third_party/onnxruntime/cmake/external/protobuf/cmake/build/install/bin/protoc ..`
5. run `emmake make export -j$(nproc)`

## Deployment to convertmodel.com

1. run `./upload_ali.sh`
