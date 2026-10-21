cmake_minimum_required(VERSION 3.24)

project(gpu_byteplane_reliability LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(bench_failsoft_pipeline src/bench_failsoft_pipeline.cu)

target_compile_options(bench_failsoft_pipeline PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)
