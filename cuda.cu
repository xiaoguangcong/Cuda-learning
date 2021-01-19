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
#define BLOCK_SIZE 16

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

__global__ static void matMultCUDA(const float* a, size_t id_a, const float *b, size_t id_b, float *c, size_t id_c, int n) {
  /*
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int idx = bid * blockDim.x + tid;
  const int row = idx / n;
  const int col = idx % n;
  int i;

  if(row < n && col < n) {
    float t = 0;
    for(i = 0; i < n; ++i) {
      t += a[row*id_a+i] * b[i*id_b+col];
    }
    c[row*id_c+col] = t;
  }
  */

  extern __shared__ float data[];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  int i, j;

  for(i = tid; i < n; i += blockDim.x) {
    data[i] = a[row * id_a + i];
  }

  __syncthreads();

  for(j = tid; j < n; j += blockDim.x) {
    float t = 0;
    float y = 0;
    for(i = 0; i < n; ++i) {
      float r;
      y -= data[i] * b[i*id_b+j];
      r = t - y;
      y = (r -t) +y;
      t = r;
    }
    c[row*id_c+j] = t;
  } 
}

__global__ static void matMultCUDAMultiBlocks(const float* a, size_t id_a, const float *b, size_t id_b, float *c, size_t id_c, int n) {
  
}

clock_t matmultCUDA(const float *a, int id_a, const float *b, int id_b, float *c, int id_c, int n) {
  float *ac, *bc, *cc;
  clock_t start, end;
  
  start = clock();
  // cudaMalloc((void**)&ac, sizeof(float) * n * n);
  // cudaMalloc((void**)&bc, sizeof(float) * n * n);
  // cudaMalloc((void**)&cc, sizeof(float) * n * n);

  // cudaMemcpy2D 函式，它是用来复制二维数组，可以指定数组的 pitch(即id_a, id_b, id_c)
  // cudaMemcpy2D(ac, sizeof(float) * n, a, sizeof(float) * id_a, sizeof(float) * n, n, cudaMemcpyHostToDevice);
  // cudaMemcpy2D(bc, sizeof(float) * n, b, sizeof(float) * id_b, sizeof(float) * n, n, cudaMemcpyHostToDevice);

  size_t pitch_a, pitch_b, pitch_c;
  //  cudaMallocPitch 的函式，可以自动以最佳的倍数来配置内存。
  cudaMallocPitch((void**)&ac, &pitch_a, sizeof(float) * n, n);
  cudaMallocPitch((void**)&bc, &pitch_b, sizeof(float) * n, n);
  cudaMallocPitch((void**)&cc, &pitch_c, sizeof(float) * n, n);

  cudaMemcpy2D(ac, pitch_a, a, sizeof(float) * id_a, sizeof(float) * n, n, cudaMemcpyHostToDevice);
  cudaMemcpy2D(bc, pitch_b, b, sizeof(float) * id_b, sizeof(float) * n, n, cudaMemcpyHostToDevice);

  // int blocks = (n + THREAD_NUM - 1) / THREAD_NUM;
  // matMultCUDA<<<blocks*n, THREAD_NUM>>>(ac, n, bc, n, cc, n, n);
  // matMultCUDA<<<n, THREAD_NUM, sizeof(float)*n>>>(ac, n, bc, n, cc, n, n);
  //matMultCUDA<<<n, THREAD_NUM, sizeof(float)*n>>>(ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float), cc, pitch_c / sizeof(float), n);

  int bx = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 blocks(bx, bx);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  matMultCUDA<<<blocks, threads>>>(ac, pitch_a/sizeof(float), bc, pitch_b / sizeof(float), cc, pitch_c / sizeof(float), n );


  //cudaMemcpy2D(c, sizeof(float) * id_c, cc, sizeof(float)*n, sizeof(float)*n, n, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(c, sizeof(float) * id_c, cc, pitch_c, sizeof(float)*n, n, cudaMemcpyDeviceToHost);

  cudaFree(ac);
  cudaFree(bc);
  cudaFree(cc);
  
  end = clock();

  return end - start;
}

bool calculateMatrixMultiply() {
  // 产生矩阵
  float *a, *b, *c, *d;
  int n = 1000;

  a = (float*)malloc(sizeof(float) * n * n);
  b = (float*)malloc(sizeof(float) * n * n);
  c = (float*)malloc(sizeof(float) * n * n);
  d = (float*)malloc(sizeof(float) * n * n);

  // 随机数产生器(种子值)
  srand(0);

  generateMatrix(a, n, n);
  generateMatrix(b, n, n);

  clock_t time = matmultCUDA(a, n, b, n, c, n, n);

  multiplyOnCPU(a, n, b, n, d, n, n);
  compareMatrixError(c, n, d, n, n);

  double sec = (double)time/CLOCKS_PER_SEC;
  printf("Time Cost: %.2f(%.2lf GFLOPS)\n", sec, 2.0*n*n*n/(sec*1E9));

  return true;

}


