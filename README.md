## Compile and install protoc 3.7.0

因为项目用到了 3.7.0 版本的 protobuf，所以需要同样版本的 protoc，我们去 protobuf 源码目录里编译安装

1. `cd third_party/protobuf/cmake`
2. `mkdir build && cd build`
3. `cmake ..`
4. `make protoc -j$(nproc)`
5. `cmake -DCMAKE_INSTALL_PREFIX=install -P cmake_install.cmake` 这一步是把编译好的文件安装到 install 目录下

## Generate MNN schema (Only do it once)

1. `cd third_party/MNN`
2. `./schema/generate.sh`

## Compile

1. install emsdk
2. run `source emsdk_env.sh`
3. `mkdir build && cd build`
4. run `emcmake cmake -DWMC_PROTOC=$PWD/../third_party/protobuf/cmake/build/install/bin/protoc ..`
5. run `emmake make export -j$(nproc)`

## Deployment to convertmodel.com

1. run `./upload_ali.sh`
