/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#include "gpu_triangle.h"
#include "triangle_methods.h"

CUDA_DEVICE bool GPUTriangle::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    return TriangleMethods::Hit(ray, v_, n_, e_, t_min, t_max, record);
}
