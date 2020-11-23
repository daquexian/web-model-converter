1. install emsdk
2. run `source /home/dev/files/repos/emsdk/emsdk_env.sh`
3. cd to build4
4. run `emcmake cmake -DWMC_PROTOC=/home/dev/files/protobuf-3.11.3/build/install/bin/protoc -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..`
5. run `emmake make export -j50`
6. run `../upload_ali.sh`

## update mnn
1. cd path/to/mnn
2. ./schema/generate.sh
