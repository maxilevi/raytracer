cmake_minimum_required(VERSION 3.10)
project(raytracer LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

file(GLOB FILES src/volumes/*.h src/volumes/*.cu src/*.cu src/*.h src/io/*.cu src/io/*.h src/math/*.h src/math/*.cu)

add_executable(raytracer ${FILES} src/gpu_tracer.cu src/gpu_tracer.h)

set_target_properties(
        raytracer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)