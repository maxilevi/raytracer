/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#include "gpu_triangle.h"

/*
 * Möller–Trumbore intersection algorithm
 *
 * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
 * */
CUDA_DEVICE bool GPUTriangle::Intersects(const Ray &ray, double &t, double& u, double &v) const
{
    const double epsilon = DOUBLE_EPSILON;
    auto edge1 = v_[1] - v_[0];
    auto edge2 = v_[2] - v_[0];
    auto h = Vector3::Cross(ray.Direction(), edge2);
    auto a = Vector3::Dot(edge1, h);
    if (a > -epsilon && a < epsilon)
        return false;
    double f = 1.0 / a;
    auto s = ray.Origin() - v_[0];
    u = f * Vector3::Dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;

    auto q = Vector3::Cross(s, edge1);
    v = f * Vector3::Dot(ray.Direction(), q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    double temp = f * Vector3::Dot(q, edge2);
    if (temp > epsilon)
    {
        t = temp;
        return true;
    }
    return false;
}

CUDA_DEVICE bool GPUTriangle::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    double t, u, v;
    if (!Intersects(ray, t, u, v)) return false;
    if (t >= t_max || t <= t_min) return false;
    record.t = t;
    record.Point = ray.Point(record.t);
    // TODO: Interpolate normals with barycentric coordinates
    record.Normal = u * n_[0] + v * n_[1] + (1 - u - v) * n_[2];
    return true;
}
