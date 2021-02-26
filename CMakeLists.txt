cmake_minimum_required(VERSION 3.10)
project(raytracer)

set(CMAKE_CXX_STANDARD 11)

file(GLOB FILES src/volumes/*.h src/volumes/*.cpp src/*.cpp src/*.h)

add_executable(raytracer ${FILES})