/*
 * Created by Maximiliano Levi on 4/9/2021.
 */

#ifndef RAYTRACER_RANDOM_H
#define RAYTRACER_RANDOM_H

#include <stdint.h>

__host__ __device__ double RandomDouble(uint32_t& seed);

__host__ __device__ int RandomInt(uint32_t& seed, int min, int max);

#endif //RAYTRACER_RANDOM_H
