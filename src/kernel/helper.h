/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#ifndef RAYTRACER_HELPER_H
#define RAYTRACER_HELPER_H

#define CUDA_DEVICE __device__
#define CUDA_HOST_DEVICE CUDA_DEVICE __host__
#if 1
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}
#else
#define CUDA_CALL(x) checkCudaErrors(x)
#endif
#include <cfloat>

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define PI 3.14159265
#define MAX_DOUBLE DBL_MAX
#define MIN_DOUBLE DBL_MIN
#define DOUBLE_EPSILON  DBL_EPSILON


#endif //RAYTRACER_HELPER_H
