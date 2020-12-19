#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
pushd build9
gzip -c -9 export.wasm > export_gz.wasm
ossutil64 --config-file ~/.ossutilconfig cp -u export_gz.wasm  oss://converter-web/export.wasm --meta=Content-Type:application/wasm#Content-Encoding:gzip
ossutil64 --config-file ~/.ossutilconfig cp -u export.js oss://converter-web/
popd
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
