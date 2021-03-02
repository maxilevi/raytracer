/*
 * Created by Maximiliano Levi on 25/02/2021.
 */

#include "triangle.h"
#include <limits>

/*
 * Möller–Trumbore intersection algorithm
 *
 * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
 * */
bool Triangle::Intersects(const Ray &ray, double &t) const
{
    double epsilon = std::numeric_limits<double>::epsilon();
    auto edge1 = v_[1] - v_[0];
    auto edge2 = v_[2] - v_[0];
    auto h = Vector3::Cross(ray.Direction(), edge2);
    auto a = Vector3::Dot(edge1, h);
    if (a > -epsilon && a < epsilon)
        return false;
    double f = 1.0 / a;
    auto s = ray.Origin() - v_[0];
    double u = f * Vector3::Dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;

    auto q = Vector3::Cross(s, edge1);
    double v = f * Vector3::Dot(ray.Direction(), q);
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

bool Triangle::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    double t;
    if (!Intersects(ray, t)) return false;
    if (t >= t_max || t <= t_min) return false;
    record.t = t;
    record.Point = ray.Point(record.t);
    // TODO: Interpolate normals with barycentric coordinates
    record.Normal = n_[0];
    return true;
}

std::ostream& operator<<(std::ostream& stream, const Triangle& triangle)
{
    stream << "(" << triangle.v_[0] << ", " << triangle.v_[1] << ", "  << triangle.v_[2] << ", ";
    stream << triangle.n_[0] << ", " << triangle.n_[1] << ", "  << triangle.n_[2] << ")";
    return stream;
}

void Triangle::Translate(Vector3 offset)
{
    for(auto & i : v_)
        i += offset;
}

void Triangle::Scale(Vector3 scale)
{
    for(auto & i : v_)
        i *= scale;
}
