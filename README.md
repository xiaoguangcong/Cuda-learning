# Cuda-learning
a project to record the cuda learning

PLATFORM：Ｕbuntu16.04
CUDA Version: 10.1
Driver Version: 430.50 

1. cuda 安装过程
参考　：　https://blog.csdn.net/jacke121/article/details/80144213

2. 笔记：

Cuda的架构下，程序分为host端和device端，Host端是在cpu执行的部分，device端是在显示芯片上执行的部分。Device端程序又称为“kernel”。


在Cuda架构下，显示芯片执行时的最小单位是 thread。 多个thread可以组成一个block。一个block中的thread能存取同一块共享的内存，而且可以快速的进行同步操作。不同block中的thread无法存取同一个共享的内存，因此无法直接互通或进行同步。

执行相同程序的block可以组成grid。

每个thread都有自己的 register 和local memory的空间。
同一个block的每个thread共享一份share memory。
所有的thread(包括不同block的thread)都共享一份global memory 、 constant memory 和站texture memory。
不同的grid则有各自的global memory 、 constant memory 和站texture memory。

最适合利用CUDA解决的问题，是可以大量并行化的问题。

CUDA 目前有两种不同的 API：Runtime API 和 Driver API，两种 API 各有其适用的范围。


