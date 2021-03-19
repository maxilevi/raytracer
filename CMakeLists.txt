cmake_minimum_required(VERSION 3.10)
project(raytracer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#set(CMAKE_CXX_FLAGS_RELEASE "/O2 /fp:fast /arch:AVX2")
#set(CMAKE_CXX_FLAGS_RELEASE "/MT")
#set(CMAKE_CXX_FLAGS_DEBUG "/MTd")

#set(CMAKE_CUDA_FLAGS_RELEASE "/MT")
#set(CMAKE_CUDA_FLAGS_DEBUG "/MTd")

file(GLOB FILES src/volumes/*.h src/volumes/*.cpp src/*.cpp src/*.h src/io/*.cpp src/io/*.h src/math/*.h src/math/*.cpp)
#file(GLOB CUDA_FILES src/*.cu src/*.cuh src/math/*.cu src/math/*.cuh)

add_executable(raytracer ${FILES} ${CUDA_FILES} src/trace.h)

#set_target_properties(
#        raytracer
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)