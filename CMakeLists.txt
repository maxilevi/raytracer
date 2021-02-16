cmake_minimum_required(VERSION 3.10)
project(raytracer)

set(CMAKE_CXX_STANDARD 11)

add_executable(raytracer src/main.cpp src/Vector3.cpp src/Vector3.h src/Ray.cpp src/Ray.h src/Sphere.cpp src/Sphere.h)