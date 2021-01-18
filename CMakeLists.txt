# CMAKE 版本最低要求
CMAKE_MINIMUM_REQUIRED(VERSION 3.8 FATAL_ERROR)

# 项目名称
project(CUDA_LEARNING C CXX CUDA)

# 设置CUDA编译器路径
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# 设置CUDA编译参数 
# -arch 指明了CUDA文件编译产生的结果所依赖的NVIDIA GPU架构的名称
# -arch= 指定前端编译目标, -code= 指定后端编译目标, compute_XX 指的是 PTX 版本, sm_XX 指的是 cubin 版本
# -rdc=true使device代码可重入
# -lcudadevrt是用来连接runtime库的，跟gcc连接库一样
set(CUDA_NVCC_FLAGS "-arch=compute_75; -code=sm_75; -rdc=true; -lcudadevrt; -std=c++14")

# 生成静态库cudas
add_library(libcuda STATIC
  cuda_util.cu
  cuda.cu
  cuda.h
  )

# 下面选项制定libcuda库使用C++14特性编译，这样所有使用libcuda库的其他可执行文件都因而使用C++14编译
target_compile_features(libcuda PUBLIC cxx_std_14)

# 生成可执行文件cuda_exe
add_executable(cuda_exe main.cu)

# nvcc把每个文件单独编译
set_property(TARGET cuda_exe 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 将可执行文件cuda_exe与cudas静态库链接在一起
target_link_libraries(cuda_exe PRIVATE libcuda)
