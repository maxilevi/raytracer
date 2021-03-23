/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#ifndef RAYTRACER_SPHERE_H
#define RAYTRACER_SPHERE_H


#include "volume.h"

class Sphere : public Volume  {
public:
    Sphere(Vector3 center, double radius) : center_(center), radius_(radius) {}
    CUDA_DEVICE bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const override;
    CUDA_DEVICE bool BoundingBox(AABB& bounding_box) const override;

    /* Accessors and mutators */
    CUDA_DEVICE inline double Radius() const { return radius_; }
    CUDA_DEVICE inline const Vector3& Center() const { return center_; }

private:
    CUDA_DEVICE bool IsValidHit(const Ray &ray, double t, double t_min, double t_max, HitResult& record) const;
    double radius_;
    Vector3 center_;
};


#endif //RAYTRACER_SPHERE_H
