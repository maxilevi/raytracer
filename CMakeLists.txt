cmake_minimum_required(VERSION 3.10)
project(raytracer LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

file(GLOB FILES src/volumes/*.h src/volumes/*.cu src/*.cu src/*.h src/io/*.cu src/io/*.h src/math/*.h src/math/*.cu src/kernel/*.h src/kernel/*.cu)

add_executable(raytracer ${FILES})

SET(CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS} -O3)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
set_target_properties(
        raytracer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)