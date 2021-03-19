/*
 * Created by Maximiliano Levi on 28/02/2021.
 */

#ifndef RAYTRACER_DEFINES_H
#define RAYTRACER_DEFINES_H

/* If we should use CUDA. If false the program uses C++17 std::execution::par_seq */
#define USE_CUDA 1

#ifdef USE_CUDA
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_DEVICE __constant__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#endif //RAYTRACER_DEFINES_H
