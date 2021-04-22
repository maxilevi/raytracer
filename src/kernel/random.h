/*
 * Created by Maximiliano Levi on 4/9/2021.
 */

#ifndef RAYTRACER_RANDOM_H
#define RAYTRACER_RANDOM_H

#include <stdint.h>
#include "helper.h"
#include "../math/vector3.h"

class Random {
public:
    static const int kRandomCount = 65536;
    static Random New(bool gpu);

    CUDA_HOST_DEVICE Random(const Random& random);
    CUDA_HOST_DEVICE ~Random();

    CUDA_HOST_DEVICE Random Reseed(uint32_t new_seed);

    CUDA_HOST_DEVICE double Double();

    CUDA_HOST_DEVICE int Int(int min, int max);

    CUDA_HOST_DEVICE Vector3 PointOnUnitSphere();

private:
    CUDA_HOST_DEVICE Random();
    static double* GetEntropy();
    bool is_in_gpu_;
    double* entropy_;
    uint32_t seed_;
    bool original_;
};



#endif //RAYTRACER_RANDOM_H
