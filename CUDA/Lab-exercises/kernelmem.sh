#!/bin/sh
#
# Written by Allan Engsig-Karup.
#
# Shell script for predicting kernel memory usage based on ptx information.
#
# first argument $1
# -Xptxas=-v

CUDA_INSTALL_PATH=/usr/local/cuda
CUDA_SDK_DIR="/Developer/GPU Computing/C"
CUDA_SDK_DIR_INC="/Developer/GPU Computing/C/common/inc"

nvcc -Xptxas=-v -I"$CUDA_INSTALL_PATH" -I"$CUDA_SDK_DIR" -I"$CUDA_SDK_DIR_INC" $1

