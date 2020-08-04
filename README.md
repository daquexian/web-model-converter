## Generate MNN schema (Only do it once)

1. cd third_party/MNN
2. ./schema/generate.sh

## Compile

1. install emsdk
2. run `source emsdk_env.sh`
3. mkdir build && cd build
4. run `emcmake cmake ..`
5. run `emmake make export -j50`

## Deployment to convertmodel.com

1. run `./upload_ali.sh`
