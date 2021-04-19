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
    if (a > -epsilon && a < epsilon)
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

CUDA_HOST_DEVICE bool TriangleMethods::Intersects3(const Ray &ray, const Vector3* vertices, const Vector3* normals, const Vector3* edges, double &t, double& u, double &v)
{
    // compute plane's normal
    auto v0 = vertices[0];
    auto v1 = vertices[1];
    auto v2 = vertices[2];

    Vector3 v0v1 = edges[0];
    Vector3 v0v2 = edges[1];
    // no need to normalize
    Vector3 N = Vector3::Cross(v0v1, v0v2); // N
    double denom = Vector3::Dot(N, N);

    // Step 1: finding P

    // check if ray and plane are parallel ?
    double NdotRayDirection = Vector3::Dot(N, ray.Direction());
    if (fabs(NdotRayDirection) < DBL_EPSILON) // almost 0
        return false; // they are parallel so they don't intersect !

    // compute d parameter using equation 2
    double d = Vector3::Dot(N, v0);

    // compute t (equation 3)
    t = (Vector3::Dot(N, ray.Origin()) + d) / NdotRayDirection;
    // check if the triangle is in behind the ray
    if (t < 0) return false; // the triangle is behind

    // compute the intersection point using equation 1
    Vector3 P = ray.Point(t);

    // Step 2: inside-outside test
    Vector3 C; // vector perpendicular to triangle's plane

    // edge 0
    Vector3 vp0 = P - v0;
    C = Vector3::Cross(v0v1, vp0);
    if (Vector3::Dot(N, C) < 0) return false;

    // edge 1
    Vector3 edge1 = v2 - v1;
    Vector3 vp1 = P - v1;
    C = Vector3::Cross(edge1, vp1);
    if ((u = Vector3::Dot(N, C)) < 0)  return false;

    // edge 2
    Vector3 edge2 = v0 - v2;
    Vector3 vp2 = P - v2;
    C = Vector3::Cross(edge2, vp2);
    if ((v = Vector3::Dot(N, C)) < 0) return false;

    u /= denom;
    v /= denom;

    return true; // this ray hits the triangle
}


CUDA_HOST_DEVICE bool TriangleMethods::Hit(const Ray &ray, const Vector3* vertices, const Vector3* normals, const Vector3* edges,
                                           const Vector3* texture_coords, const Material* material, double t_min, double t_max, HitResult &record)
{
    double t, u, v;
    if (!Intersects3(ray, vertices, normals, edges, t, u, v)) return false;
    if (t >= t_max || t <= t_min) return false;
    double r =  (1.0 - u - v);
    record.t = t;
    record.u = u;
    record.v = v;
    record.Point = ray.Point(record.t);
    record.Normal = Vector3::Cross(edges[0], edges[1]);//u * normals[0] + v * normals[1] + r * normals[2];
    double coord0 = u * texture_coords[0][0] + v * texture_coords[0][1] + r * texture_coords[0][2];
    double coord1 = u * texture_coords[1][0] + v * texture_coords[1][1] + r * texture_coords[1][2];
    record.Color = material->Sample(coord0, coord1);
    //if (record.Color.X() <= DBL_EPSILON && record.Color.Y() <= DBL_EPSILON && record.Color.Z() <= DBL_EPSILON)
    //    record.Color = Vector3(1, 0, 0);//printf("error %f %f %f, %f %f\n", u, v, r, coord0, coord1);
    return true;
}
