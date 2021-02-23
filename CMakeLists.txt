cmake_minimum_required(VERSION 3.10)
project(raytracer)

set(CMAKE_CXX_STANDARD 11)

file(GLOB CPP_FILES src/*.cpp)
file(GLOB H_FILES src/*.h)

add_executable(raytracer ${CPP_FILES} ${H_FILES})