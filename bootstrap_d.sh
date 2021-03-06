#!/bin/bash

export PREFIX=$HOME/devroot

export SCRIPT_DIR=`dirname $0`;
export CMAKE_PREFIX_PATH=$PREFIX:$CMAKE_PREFIX_PATH
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig/:$PKG_CONFIG_PATH
cmake \
    -G "Eclipse CDT4 - Ninja" \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    $SCRIPT_DIR

