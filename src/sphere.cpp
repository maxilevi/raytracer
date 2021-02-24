/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#include "Sphere.h"

bool Sphere::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
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
    double b = 2.0 * Vector3::Dot(oc, ray.Direction());
    double c = Vector3::Dot(oc, oc) - radius_*radius_;
    double discriminant = b*b - 4*a*c;
    std::cout << discriminant << std::endl;
    if (discriminant > 0)
    {
        double temp = (-b - sqrt(b * b - a * c)) / a;
        if (temp < t_max && temp > t_min) {
            record.t = temp;
            record.Point = ray.Point(record.t);
            record.Normal = (record.Point - this->Center()) / radius_;
            return true;
        }
        temp = (-b + sqrt(b * b - a * c)) / a;
        if (temp < t_max && temp > t_min) {
            record.t = temp;
            record.Point = ray.Point(record.t);
            record.Normal = (record.Point - this->Center()) / radius_;
            return true;
        }
    }
    return false;
}
