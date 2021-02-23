/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#ifndef RAYTRACER_DRAWABLE_OBJECT_H
#define RAYTRACER_DRAWABLE_OBJECT_H


#include "vector3.h"
#include "ray.h"

class IDrawableObject {
public:
    virtual bool Hit(const Ray& ray, double t_min, double t_max, ) const = 0;
};


#endif //RAYTRACER_DRAWABLE_OBJECT_H
