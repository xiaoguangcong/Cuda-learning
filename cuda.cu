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
#define THREAD_NUM 256
#define BLOCK_NUM 32

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

// 计算平方和的函式（多个线程）
__global__ static void sumOfSquaresParallel(int *num, int *result, clock_t *time) {
  const int tid = threadIdx.x;
  int sum = 0;
  int i;
  clock_t start;
  if(tid == 0) { start = clock(); }
  for(i = tid; i < DATA_SIZE; i = i + THREAD_NUM) {
    sum += num[i] * num[i];
  }

  result[tid] = sum;
  if(tid == 0) *time = clock() - start;
}

// 计算平方和的函式（多个块）
__global__ static void sumOfSquaresMultiBlocks(int *num, int *result, clock_t *time) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  int sum = 0;
  int i;
  if(tid == 0) { time[bid] = clock(); }
  for(i = bid * THREAD_NUM + tid; i < DATA_SIZE; i = i + BLOCK_NUM * THREAD_NUM) {
    sum += num[i] * num[i];
  }

  result[bid * THREAD_NUM + tid] = sum;
  if(tid == 0) time[bid + BLOCK_NUM] = clock();
}

// 计算平方和的函式（加上Thread的同步）
__global__ static void sumOfSquaresSync(int *num, int *result, clock_t *time) {
  // 声明变量是shared memory，是一个block中的每个thread都共享的内存。
  extern __shared__ int shared[];
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  int i;
  int offset = THREAD_NUM >> 1;
  if(tid == 0) { time[bid] = clock(); }
  shared[tid] = 0;
  for(i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM) {
    shared[tid] += num[i] * num[i];
  }

  // block中所有的thread都要同步到这个点，才能继续执行。
  __syncthreads();
  /***
  if(tid == 0) {
    for(int i = 1; i < THREAD_NUM; ++i) {
      shared[0] += shared[i];
    }
    result[bid] = shared[0];
  }
  ***/
  while(offset > 0) {
    if(tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    offset >>= 1;
    __syncthreads();
  }

  if(tid == 0) {
    result[bid] = shared[0];
    time[bid + BLOCK_NUM] = clock();
  }
}

void calculateOnCPU(int *data) {
  // 在cpu端直接计算各个数的平方和
  int final_sum = 0;
  for(int i = 0; i < DATA_SIZE; ++i) {
    final_sum += data[i] * data[i];
  }
  printf("sum(CPU): %d\n", final_sum);
}

void calculateOnGPU(int *data) {
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
  cudaMemcpy(&time_cost, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
  cudaFree(gpu_data);
  cudaFree(result);
  cudaFree(time);

  printf("sum(GPU): %d time cost: %ld\n", sum, time_cost);
}

void calculateUsingMultiThreads(int *data) {

  int *gpu_data, *result;
  clock_t *time;
  cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
  cudaMalloc((void **)&result, sizeof(int) * THREAD_NUM);
  cudaMalloc((void **)&time, sizeof(clock_t));

  cudaMemcpy(gpu_data, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

  // 函式名称中标明线程数量
  sumOfSquaresParallel<<<1, THREAD_NUM, 0>>>(gpu_data, result, time);

  int sum[THREAD_NUM];
  clock_t time_cost;
  cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(&time_cost, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
  cudaFree(gpu_data);
  cudaFree(result);
  cudaFree(time);

  // 在cpu端把个部分数据的平方和进行加总
  int final_sum = 0;
  for(int i = 0; i < THREAD_NUM; ++i) {
    final_sum += sum[i];
  }
  printf("sum(GPU, multi threads): %d time: %ld\n", final_sum, time_cost);
}

void calculateUsingMultiBlocks(int *data) {

  int *gpu_data, *result;
  clock_t *time;
  cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
  cudaMalloc((void **)&result, sizeof(int) * THREAD_NUM * BLOCK_NUM);
  cudaMalloc((void **)&time, sizeof(clock_t) * BLOCK_NUM * 2);

  cudaMemcpy(gpu_data, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

  // 函式名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
  sumOfSquaresMultiBlocks<<<BLOCK_NUM, THREAD_NUM, 0>>>(gpu_data, result, time);

  int sum[THREAD_NUM  * BLOCK_NUM];
  clock_t time_cost[BLOCK_NUM * 2];
  cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(&time_cost, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
  cudaFree(gpu_data);
  cudaFree(result);
  cudaFree(time);

  // 在cpu端把个部分数据的平方和进行加总
  int final_sum = 0;
  for(int i = 0; i < THREAD_NUM * BLOCK_NUM; ++i) {
    final_sum += sum[i];
  }

  clock_t min_start, max_end;
  min_start = time_cost[0];
  max_end = time_cost[BLOCK_NUM];
  for(int i = 0; i < BLOCK_NUM; ++i) {
    if(min_start > time_cost[i]) {
      min_start = time_cost[i];
    }
    if(max_end < time_cost[i + BLOCK_NUM]) {
      max_end = time_cost[i + BLOCK_NUM];
    }
  }
  printf("sum(GPU, multi blocks): %d time: %ld\n", final_sum, max_end - min_start);
}

void calculateUsingSync(int *data) {
  int *gpu_data, *result;
  clock_t *time;
  cudaMalloc((void **)&gpu_data, sizeof(int) * DATA_SIZE);
  cudaMalloc((void **)&result, sizeof(int) * BLOCK_NUM); 
  cudaMalloc((void **)&time, sizeof(clock_t) * BLOCK_NUM * 2);

  cudaMemcpy(gpu_data, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);
  
  // 函式名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
  sumOfSquaresSync<<<BLOCK_NUM, THREAD_NUM, THREAD_NUM * sizeof(int)>>>(gpu_data, result, time);

  int sum[BLOCK_NUM];
  clock_t time_cost[BLOCK_NUM * 2];
  cudaMemcpy(&sum, result, sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost);
  cudaMemcpy(&time_cost, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
  cudaFree(gpu_data);
  cudaFree(result);
  cudaFree(time);

  int final_sum = 0;
  for(int i = 0; i < BLOCK_NUM; ++i) {
    final_sum += sum[i];
  } 

  clock_t min_start, max_end;
  min_start = time_cost[0];
  max_end = time_cost[BLOCK_NUM];
  for(int i = 0; i < BLOCK_NUM; ++i) {
    if(min_start > time_cost[i]) {
      min_start = time_cost[i];
    }
    if(max_end < time_cost[i + BLOCK_NUM]) {
      max_end = time_cost[i + BLOCK_NUM];
    }
  }
  printf("sum(GPU, Sync): %d time: %ld\n", final_sum, max_end - min_start);
}



bool calculateSumOfSquares() {
  // 产生随机数
  int data[DATA_SIZE];
  generateNumbers(data, DATA_SIZE);

  // 在CPU上计算
  calculateOnCPU(data);

  // 在GPU上计算，one thread， one block
  calculateOnGPU(data);

  // 在GPU上计算，multi threads， one block
  calculateUsingMultiThreads(data);

  // 在GPU上计算，multi threads， multi blocks
  calculateUsingMultiBlocks(data);

  // 在GPU上计算，multi threads， multi blocks, using sync
  calculateUsingSync(data);

  return true;
}


