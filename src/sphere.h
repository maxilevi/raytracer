/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#ifndef RAYTRACER_SPHERE_H
#define RAYTRACER_SPHERE_H


#include "volume.h"

class Sphere : public Volume  {
public:
    Sphere(Vector3 center, double radius) : center_(center), radius_(radius) {}
    bool Hit(const Ray& ray, double t_min, double t_max, VolumeHit& record) const override;

    /* Accessors and mutators */
    inline double Radius() const { return radius_; }
    inline Vector3 Center() const { return center_; }

private:
    double radius_;
    Vector3 center_;
};


#endif //RAYTRACER_SPHERE_H
