/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#ifndef RAYTRACER_TRIANGLE_METHODS_H
#define RAYTRACER_TRIANGLE_METHODS_H
#include "../math/ray.h"
#include "../kernel/helper.h"
#include "../materials/material.h"
#include "hit_result.h"

class TriangleMethods {
public:
    CUDA_HOST_DEVICE static bool Hit(const Ray&, const Vector3*, const Vector3*, const Vector3*, const Vector3*, const Material*, double, double, HitResult&);
private:
    CUDA_HOST_DEVICE static bool Intersects(const Ray&, const Vector3*, const Vector3*, const Vector3*, double &, double& , double &);
    CUDA_HOST_DEVICE static bool Intersects3(const Ray&, const Vector3*, const Vector3*, const Vector3*, double &, double&, double &);
};


#endif //RAYTRACER_TRIANGLE_METHODS_H
