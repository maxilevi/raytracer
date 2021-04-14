/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#include "triangle_methods.h"
#define CULL_BACKFACE 0

/*
 * Möller–Trumbore intersection algorithm
 *
 * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
 * */
CUDA_HOST_DEVICE bool TriangleMethods::Intersects(const Ray &ray, const Vector3* vertices, const Vector3* normals, const Vector3* edges, double &t, double& u, double &v)
{
#if CULL_BACKFACE
    const double epsilon = DOUBLE_EPSILON;
    auto edge1 = vertices[1] - vertices[0];
    auto edge2 = vertices[2] - vertices[0];
    auto h = Vector3::Cross(ray.Direction(), edge2);
    auto det = Vector3::Dot(edge1, h);
    if (det < epsilon)
        return false;

    auto tvec = (ray.Origin(), vertices[0]);
    u = Vector3::Dot(tvec, h);

    if (u < 0.0 || u > det)
        return false;

    auto q = Vector3::Cross(tvec, edges[0]);
    v = Vector3::Dot(ray.Direction(), q);

    if (v < 0.0 || u + v > det)
        return false;

    t = Vector3::Dot(vertices[1], q);
    auto inv_det = 1.0 / det;
    t *= inv_det;
    u *= inv_det;
    v *= inv_det;
    return true;
#else
    const double epsilon = DOUBLE_EPSILON;
    auto h = Vector3::Cross(ray.Direction(), edges[1]);
    auto a = Vector3::Dot(edges[0], h);
    if (/*a > -epsilon &&*/ a < epsilon)
        return false;
    double f = 1.0 / a;
    auto s = ray.Origin() - vertices[0];
    u = f * Vector3::Dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;

    auto q = Vector3::Cross(s, edges[0]);
    v = f * Vector3::Dot(ray.Direction(), q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    double temp = f * Vector3::Dot(q, edges[1]);
    if (temp > epsilon)
    {
        t = temp;
        return true;
    }
    return false;
#endif
}

CUDA_HOST_DEVICE bool TriangleMethods::Intersects2(const Ray &ray, const Vector3* vertices, const Vector3* normals, const Vector3* edges, double &t, double& u, double &v)
{
    /*
     * ray = P0 + D * t
     * plane = P . N + d = 0
     * = (P0 + D * t) . N + d = 0
     * => t = -(P0 . N + d) / (D . N)
     * */
    Vector3 normal = normals[0];
    double d = Vector3::Dot(normal, vertices[0]);

    t = -(Vector3::Dot(ray.Origin(), normal) + d) / Vector3::Dot(ray.Direction(), normal);
    Vector3 p = ray.Point(t);
    //for(int i = 0; i < 3; ++i)
    {
        auto v1 = vertices[0] - p;
        auto v2 = vertices[1] - p;
        auto n1 = Vector3::Cross(v1, v2);
        auto d1 = -Vector3::Dot(ray.Origin(), n1);
        if (Vector3::Dot(p, n1) + d1 < 0)
            return false;
    }
    return true;
}

CUDA_HOST_DEVICE bool TriangleMethods::Hit(const Ray &ray, const Vector3* vertices, const Vector3* normals, const Vector3* edges,
                                           double t_min, double t_max, HitResult &record)
{
    double t, u, v;
    if (!Intersects(ray, vertices, normals, edges, t, u, v)) return false;
    if (t >= t_max || t <= t_min) return false;
    record.t = t;
    record.Point = ray.Point(record.t);
    // TODO: Interpolate normals with barycentric coordinates
    record.Normal = normals[0];//u * normals[0] + v * normals[1] + (1 - u - v) * normals[2];
    return true;
}
