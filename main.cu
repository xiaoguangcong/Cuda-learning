/*
 *   Copyright (c) 2021 
 *   All rights reserved.
 */
#include "cuda.h"

int main(){
    if(!InitCUDA()) {
        return 0;
    }

    printf("CUDA initialized.\n");
    return 0;
}