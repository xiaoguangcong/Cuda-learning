/*
 *   Copyright (c) 2021 
 *   All rights reserved.
 */
#include "cuda.h"
#include "cuda_util.cu"
#include <stdio.h>
#include <stdbool.h>

bool InitCUDA() {
    int count = 0;
    
    // 可以通过 cudaGetDeviceCount 函数获取 CUDA 的设备数, 函数通过引用传递 count 值，获取当前支持的 CUDA 设备数。
    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "This is no device.\n");
        return false;
    }

    int i;
    for(i = 0; i < count; ++i) {
        // 可以通过 cudaGetDeviceProperties 函数获取 CUDA 设备的属性
        struct cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                printDeviceProp(prop);
                break;
            }
        }
    }

    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x .\n");
        return false;
    }

    // 设置某一块的Device作为这个主机host上某一个运行线程的设备，这个函数必须在使用_global_的函数或者Runtime的其他API调用之前才能生效
    cudaSetDevice(i);

    return true;
}