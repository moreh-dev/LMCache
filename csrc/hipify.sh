#!/bin/bash

rm -rf hip_output
mkdir hip_output
python hipify.py -p . -o hip_output ac_dec.cu ac_enc.cu cachegen_kernels.cuh cal_cdf.cu mem_kernels.cu  mem_kernels.cuh pybind.cpp
