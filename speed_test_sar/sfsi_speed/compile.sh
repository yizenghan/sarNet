#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo "Building dcn..."
cd mmcls/ops/dcn
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace