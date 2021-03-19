/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#ifndef RAYTRACER_DRAWABLE_OBJECT_H
#define RAYTRACER_DRAWABLE_OBJECT_H


#include "../math/vector3.h"
#include "../math/ray.h"
#include "aabb.h"

struct HitResult
{
    CUDA_CALLABLE_MEMBER HitResult() {};
    double t = 0;
    Vector3 Point;
    Vector3 Normal;
};

class Volume {
public:
    CUDA_CALLABLE_MEMBER virtual bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const = 0;
    CUDA_CALLABLE_MEMBER virtual bool BoundingBox(AABB& bounding_box) const = 0;
};


#endif //RAYTRACER_DRAWABLE_OBJECT_H
