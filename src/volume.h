/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#ifndef RAYTRACER_DRAWABLE_OBJECT_H
#define RAYTRACER_DRAWABLE_OBJECT_H


#include "vector3.h"
#include "ray.h"

struct HitResult
{
    float t = 0;
    Vector3 point;
    Vector3 normal;
};

class Volume {
public:
    virtual bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const = 0;
};


#endif //RAYTRACER_DRAWABLE_OBJECT_H
