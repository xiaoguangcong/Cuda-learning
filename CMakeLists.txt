# CMAKE 版本最低要求
CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# 项目名称
project(CUDA_LEARNING C CXX CUDA)

set(CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20; -rdc=true; -lcudadevrt")

FIND_PACKAGE(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project built failes!")
endif()

# 项目名称
project(CUDA_LEARNING C CXX CUDA)

# 设置CUDA编译器路径
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# 设置CUDA编译参数
set(CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20; -rdc=true; -lcudadevrt")

# 生成静态库cudas
add_library(cudas STATIC
  cuda.cu
  cuda.h
  )

target_compile_features(cudas PUBLIC cxx_std_11)

set_target_properties( cudas
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 生成可执行文件cuda_exe
add_executable(cuda_exe main.cu)

# 
set_property(TARGET cuda_exe 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 将可执行文件cuda_exe与cudas静态库链接在一起
target_link_libraries(cuda_exe PRIVATE cudas)
