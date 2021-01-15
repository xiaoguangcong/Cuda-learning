/*
 *   Copyright (c) 2021 
 *   All rights reserved.
 */
#include "cuda.h"
#include <stdio.h>
#include <stdbool.h>

bool InitCUDA() {
    int count = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "This is no device.\n");
        return false;
    }

    int i;
    for(i = 0; i < count; ++i) {
        struct cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }

    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x .\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}