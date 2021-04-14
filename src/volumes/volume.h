/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#ifndef RAYTRACER_DRAWABLE_OBJECT_H
#define RAYTRACER_DRAWABLE_OBJECT_H


#include "../math/vector3.h"
#include "../math/ray.h"
#include "aabb.h"
#include "hit_result.h"

class Volume {
public:
    virtual bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const = 0;
    virtual bool BoundingBox(AABB& bounding_box) const = 0;
};


#endif //RAYTRACER_DRAWABLE_OBJECT_H
