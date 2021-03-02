cmake_minimum_required(VERSION 3.10)
#project(raytracer LANGUAGES CUDA CXX)
project(raytracer LANGUAGES CXX)

#set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.24.28314/bin/Hostx64/x64/cl.exe")
#find_package(CUDA)
#set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

file(GLOB FILES src/volumes/*.h src/volumes/*.cpp src/*.cpp src/*.h src/io/*.cpp src/io/*.h)
#file(GLOB CUDA_FILES src/*.cu src/*.cuh)

add_executable(raytracer ${FILES} )
#add_executable(raytracer ${FILES} ${CUDA_FILES})

#set_target_properties(
#        raytracer
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)