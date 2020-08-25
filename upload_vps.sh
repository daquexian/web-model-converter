#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
pushd build4
# gzip -c -9 export.wasm > export_gz.wasm
# scp export_gz.wasm daquexian@45.32.53.179:/home/daquexian/web_content/export.wasm
scp export.wasm daquexian@45.32.53.179:/home/daquexian/web_content/export.wasm
scp export.js daquexian@45.32.53.179:/home/daquexian/web_content/
popd
pushd web/
scp index.html robots.txt daquexian@45.32.53.179:/home/daquexian/web_content/
popd
