cmake_minimum_required(VERSION 3.10)
project(raytracer LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

file(GLOB HEADER_FILES src/volumes/*.h src/*.h src/io/*.h src/math/*.h src/kernel/*.h src/backends/*.h)
file(GLOB CUDA_FILES src/volumes/*.cu src/*.cu src/io/*.cu src/math/*.cu src/kernel/*.cu src/backends/*.cu)
file(GLOB CPP_FILES src/volumes/*.cpp src/*.cpp src/io/*.cpp src/math/*.cpp src/kernel/*.cpp src/backends/*.cpp)
message(STATUS "${CUDA_FILES}")
add_executable(raytracer ${HEADER_FILES} ${CUDA_FILES} ${CPP_FILES} src/volumes/triangle_methods.cu src/volumes/triangle_methods.h)

#SET(CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS} -O3)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
#target_compile_options(raytracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_options(raytracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3,-v, -O3>)
set_target_properties(
        raytracer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)