cmake_minimum_required(VERSION 3.10)
project(raytracer LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

file(GLOB FILES src/volumes/*.h src/volumes/*.cu src/*.cu src/*.h src/io/*.cu src/io/*.h src/math/*.h src/math/*.cu src/kernel/*.h src/kernel/*.cu)
message(STATUS "${CUDA_FILES}")
add_executable(raytracer ${FILES} src/helper.h src/helper.cu)

#SET(CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS} -O3)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
#target_compile_options(raytracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_compile_options(raytracer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3,-v, -O3>)
set_target_properties(
        raytracer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)