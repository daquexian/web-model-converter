#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
pushd build4
gzip -c -9 export.wasm > export_gz.wasm
# sed -i "1442iwasmBinaryFile = 'https://convertmodel-1256200149.file.myqcloud.com/export_gz.wasm';" export.js
coscmd upload export_gz.wasm /export.wasm -H "{'Content-Type': 'application/wasm', 'Content-Encoding': 'gzip'}"
coscmd upload export.js /
popd
pushd web/
coscmd upload index.html /
coscmd upload robots.txt /
# tcb hosting:deploy
# scp * daquexian@139.155.88.144:/var/www/html/test
# scp * daquexian@139.155.88.144:/var/www/html
popd
