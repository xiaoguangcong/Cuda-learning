/*
 *   Copyright (c) 2021
 *   All rights reserved.
 */
#include "cuda.h"
#include "cuda_util.cu"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define DATA_SIZE 1048576

bool initCUDA() {
  int count = 0;

  // 可以通过 cudaGetDeviceCount 函数获取 CUDA 的设备数, 函数通过引用传递 count
  // 值，获取当前支持的 CUDA 设备数。
  cudaGetDeviceCount(&count);
  if (count == 0) {
    fprintf(stderr, "This is no device.\n");
    return false;
  }

  int i;
  for (i = 0; i < count; ++i) {
    // 可以通过 cudaGetDeviceProperties 函数获取 CUDA 设备的属性
    struct cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      if (prop.major >= 1) {
        printDeviceProp(prop);
        break;
      }
    }
  }

  if (i == count) {
    fprintf(stderr, "There is no device supporting CUDA 1.x .\n");
    return false;
  }

  // 设置某一块的Device作为这个主机host上某一个运行线程的设备，这个函数必须在使用_global_的函数或者Runtime的其他API调用之前才能生效
  cudaSetDevice(i);

  return true;
}

// 计算平方和的函式
// CUDA 提供了一个 clock 函式，可以取得目前的
// timestamp，很适合用来判断一段程序执行所花费的时间（单位为 GPU
// 执行单元的频率）。
__global__ static void sumOfSquares(int *num, int *result, clock_t *time) {
  int sum = 0;
  int i;
  clock_t start = clock();
  for (i = 0; i < DATA_SIZE; ++i) {
    sum += num[i] * num[i];
  }

  *result = sum;
  *time = clock() - start;
}

bool calculateSumOfSquares() {
  // 产生随机数
  int data[DATA_SIZE];
  generateNumbers(data, DATA_SIZE);

  // 在显卡中开辟内存，并将数据从主存中复制到显卡内存
  int *gpu_data, *result;
  clock_t *time;
  cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
  cudaMalloc((void **)&result, sizeof(int));
  cudaMalloc((void **)&time, sizeof(clock_t));

  // cudaMemcpy 将产生的随机数复制到显卡内存中
  // cudaMemcpyHostToDevice - 从内存复制到显卡内存
  // cudaMemcpyDeviceToHost - 从显卡内存复制到内存
  cudaMemcpy(gpu_data, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

  // 函式名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
  sumOfSquares<<<1, 1, 0>>>(gpu_data, result, time);

  int sum;
  clock_t time_cost;
  cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&time_cost, time, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(gpu_data);
  cudaFree(result);
  cudaFree(time);

  printf("sum of squares: %d time cost: %ld\n", sum, time_cost);

  return true;
}
