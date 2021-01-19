/*
 *   Copyright (c) 2021
 *   All rights reserved.
 */
#include "cuda.h"

void printDeviceProp(const cudaDeviceProp &prop) {
  /***
  struct cudaDeviceProp {
      char name [256];          // 用于标识设备的ASCII字符串
      size_t totalGlobalMem;    // 设备上可用的全局存储器的总量，以字节为单位
      size_t shareMemPerBlock;  //
  线程块可以使用的共享存储器的最大值，以字节为单位;多存储器上的所有线程块可以同时共享这些存储器;
      int regsPerBlock;         //
  线程块可以使用的32位寄存器的最大值;多存储器上的所有线程块可以同时共享这些寄存器;
      int warpSize;             // 按线程计算的warp块的大小
      size_t memPitch;          //
  允许通过cudaMallocPitch()为包含存储器区域的存储器复制函数分配的最大间距(pitch)，以字节为单位
      int maxThreadsPerBlock;   // 每个块中最大线程数
      int maxThreadsDim [3];    // 块中各个维度的最大值
      int maxGridSize [3];      // 网格中各个维度的最大值
      size_t totalConstMem;     // 设备上可用的不变存储器总量，以字节为单位
      int major;                // 定义设备计算能力的主要修订号
      int minor;                // 定义设备计算能力的次要修订号
      int clockRate;            // 以千赫为单位的时钟频率
      size_t textureAlignment;  //
  对齐要求，与textureAligment字节对齐的纹理基址无需对纹理采样应用偏移
      int deviceOverlap;        //
  如果设备可在主机和设备之间并发复制存储器，同时又能执行内核，则此值为1,否则为0
      int multiProcessCount;    // 设备上多处理器的数量
  };
  ***/

  printf("Device Name : %s\n", prop.name);
  printf("totalGlobalMemory : %ld\n", prop.totalGlobalMem);
  printf("sharedMemoryPerBlock : %ld\n", prop.sharedMemPerBlock);
  printf("regsPerBlock : %d\n", prop.regsPerBlock);
  printf("warpSize : %d\n", prop.warpSize);
  printf("memPitch : %ld\n", prop.memPitch);
  printf("maxThreadsPerBlock : %d\n", prop.maxThreadsPerBlock);
  printf("maxThreadsDim[0 - 2] : %d %d %d\n", prop.maxThreadsDim[0],
         prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0],
         prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("totalConstMem : %ld\n", prop.totalConstMem);
  printf("major.minor : %d %d\n", prop.major, prop.minor);
  printf("clockRate : %d\n", prop.clockRate);
  printf("textureAlignment : %ld\n", prop.textureAlignment);
  printf("deviceOverlap : %d\n", prop.deviceOverlap);
  printf("multiProcessorCount : %d\n", prop.multiProcessorCount);
}

// 生成0-9的随机数
void generateNumbers(int *number, int size) {
  for (int i = 0; i < size; ++i) {
    number[i] = rand() % 10;
  }
}

// 用 double 来储存暂时的计算结果，以提高精确度。
void multiplyOnCPU(const float *a, int id_a, const float *b, int id_b, float *c, int id_c, int n) {
  int i, j, k;

  for(i = 0; i < n; ++i) {
    for(j = 0; j < n; ++j) {
      double t = 0;
      for(k = 0; k < n; ++k) {
        t += a[i * id_a + k] * b[k * id_b + j];
      }
      c[i*id_c+j] = t;
    }
  }
}



// 利用随机数把矩阵生成 0 - 1 的数
// C语言中无法声明变动大小的二维矩阵，所有使用 i*id_a+j 的方式
void generateMatrix(float *a, int id_a, int n) {
  int i, j;

  for(i = 0; i < n; ++i) {
    for(j = 0; j < n; ++j) {
      a[i*id_a+j] = (float)rand()/RAND_MAX + (float)rand()/(RAND_MAX * RAND_MAX);
    }
  }
}

// 式计算两个矩阵的最大相对误差和平均相对误差，并把结果印出来。
void compareMatrixError(const float *a, int id_a, const float *b, int id_b, int n) {
  float max_err = 0;
  float average_err = 0;
  int i, j;

  for(i = 0; i < n; ++i) {
    for(j = 0; j < n; ++j) {
      if(b[i*id_b+j] != 0) {
        float err = fabs((a[i*id_a+j] - b[i*id_b+j])/b[i*id_b+j]);
        if(max_err < err) {
          max_err = err;
        }
        average_err += err;
      }
    }
  }

  printf("Max error: %g Average error: %g\n", max_err, average_err/(n*n));
}
