/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#ifndef RAYTRACER_SPHERE_H
#define RAYTRACER_SPHERE_H


#include "volume.h"

class Sphere : public Volume  {
public:
    Sphere(Vector3 center, double radius) : center_(center), radius_(radius) {}
    CUDA_CALLABLE_MEMBER bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const override;
    CUDA_CALLABLE_MEMBER bool BoundingBox(AABB& bounding_box) const override;

    /* Accessors and mutators */
    CUDA_CALLABLE_MEMBER inline double Radius() const { return radius_; }
    CUDA_CALLABLE_MEMBER inline const Vector3& Center() const { return center_; }

private:
    CUDA_CALLABLE_MEMBER bool IsValidHit(const Ray &ray, double t, double t_min, double t_max, HitResult& record) const;
    double radius_;
    Vector3 center_;
};


#endif //RAYTRACER_SPHERE_H
