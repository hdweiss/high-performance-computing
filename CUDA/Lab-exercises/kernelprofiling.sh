#!/bin/sh
#
# Written by Allan Engsig-Karup.
#
# Shell script for activitating the text version of the CUDA profiler.
#
# Note: Only works if the nvcc compiler debug flag "-g" 
# has been invoked during compilation.
# 
# At command prompt type
# $ source kernelprofiling.sh
#
# to have these environmental variables defined.

export CUDA_PROFILE=1
export CUDA_PROFILE_CONFIG=profile_config

