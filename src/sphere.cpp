/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#include "Sphere.h"

bool Sphere::Hit(const Ray &ray, double t_min, double t_max, VolumeHit &record) const
{
    auto offset = ray.Origin() - this->Center();
    /* TODO: quadratic equation */
    return (discriminant > 0);
}
