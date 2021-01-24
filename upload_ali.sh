#!/usr/bin/env bash

set -e -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
pushd build9
gzip -c -9 export.wasm > export_gz.wasm
ossutil64 --config-file ~/.ossutilconfig cp -u export_gz.wasm  oss://converter-web/export.wasm --meta=Content-Type:application/wasm#Content-Encoding:gzip
ossutil64 --config-file ~/.ossutilconfig cp -u export.js oss://converter-web/
popd
# pushd /home/dev/files/repos/web-model-converter/third_party/ncnn/build/tools/caffe
# gzip -c -9 caffe2ncnn.wasm > caffe2ncnn_gz.wasm
# ossutil64 --config-file ~/.ossutilconfig cp -u caffe2ncnn_gz.wasm  oss://converter-web/caffe2ncnn.wasm --meta=Content-Type:application/wasm#Content-Encoding:gzip
# ossutil64 --config-file ~/.ossutilconfig cp -u caffe2ncnn.js oss://converter-web/
# popd
function upload_x2ncnn {
  pushd $1
  js=$2.js
  wasm=$2.wasm
  zipped_wasm=$2_gz.wasm
  gzip -c -9 $wasm > $zipped_wasm
  ossutil64 --config-file ~/.ossutilconfig cp -u $zipped_wasm  oss://converter-web/$wasm --meta=Content-Type:application/wasm#Content-Encoding:gzip
  ossutil64 --config-file ~/.ossutilconfig cp -u $js oss://converter-web/
  popd
}
upload_x2ncnn /home/dev/files/repos/web-model-converter/third_party/ncnn/build/tools/caffe caffe2ncnn
upload_x2ncnn /home/dev/files/repos/web-model-converter/third_party/ncnn/build/tools/darknet darknet2ncnn
pushd ./onnxopt/build
gzip -c -9 export_onnxopt.wasm > export_onnxopt_gz.wasm
ossutil64 --config-file ~/.ossutilconfig cp -u export_onnxopt_gz.wasm  oss://converter-web/export_onnxopt.wasm --meta=Content-Type:application/wasm#Content-Encoding:gzip
ossutil64 --config-file ~/.ossutilconfig cp -u export_onnxopt.js oss://converter-web/
popd
pushd web/
ossutil64 --config-file ~/.ossutilconfig cp -u index.html oss://converter-web/
ossutil64 --config-file ~/.ossutilconfig cp -u convert.js oss://converter-web/
ossutil64 --config-file ~/.ossutilconfig cp -u ui.js oss://converter-web/
ossutil64 --config-file ~/.ossutilconfig cp -u robots.txt oss://converter-web/
popd
