/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#include "sphere.h"

CUDA_DEVICE bool Sphere::IsValidHit(const Ray &ray, double t, double t_min, double t_max, HitResult& record) const
{
    if (t >= t_max || t <= t_min) return false;
    record.t = t;
    record.Point = ray.Point(record.t);
    record.Normal = (record.Point - this->Center()) / radius_;
    return true;
}

CUDA_DEVICE bool Sphere::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    /*
     * dot(p(t)-c, p(t)-c) = r^2 -> Equation for finding the points the ray collides
     * dot(A + B * t - C, A + B * t - C) = r^2
     * A^2 + A * B * t - AC + A * B * t + (B * t)^2 - C * B * t
     *
     *
     * */
    auto oc = ray.Origin() - this->Center();
    double a = Vector3::Dot(ray.Direction(), ray.Direction());
    double b = Vector3::Dot(oc, ray.Direction());
    double c = Vector3::Dot(oc, oc) - radius_ * radius_;
    double discriminant = b*b - a*c;
    if (discriminant > 0)
    {
        double temp = (-b - sqrt(b * b - a * c)) / a;
        if (IsValidHit(ray, temp, t_min, t_max, record))
            return true;

        temp = (-b + sqrt(b * b - a * c)) / a;
        if (IsValidHit(ray, temp, t_min, t_max, record))
            return true;
    }
    return false;
}

CUDA_DEVICE bool Sphere::BoundingBox(AABB &bounding_box) const
{
    bounding_box = AABB(center_ - Vector3(radius_), center_ + Vector3(radius_));
    return true;
}
