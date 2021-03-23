/*
 * Created by Maximiliano Levi on 28/02/2021.
 */

#ifndef RAYTRACER_DEFINES_H
#define RAYTRACER_DEFINES_H

/* If we should use CUDA. If false the program uses C++17 std::execution::par_seq */
#define USE_CUDA 1

#ifdef USE_CUDA
#define CUDA_DEVICE __host__ __device__
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}
#else
#define CUDA_DEVICE
#endif

#endif //RAYTRACER_DEFINES_H
