/*
 * Created by Maximiliano Levi on 25/02/2021.
 */

#include "triangle.h"
#include "triangle_methods.h"
#include <limits>
#include <algorithm>

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

bool Triangle::BoundingBox(AABB &bounding_box) const
{
    Vector3 min(std::numeric_limits<double>::max()), max(std::numeric_limits<double>::min());
    for(auto& v : v_)
    {
        for(int i = 0; i < 3; ++i)
        {
            min[i] = std::min(v[i], min[i]);
            max[i] = std::max(v[i], max[i]);
        }
    }
    bounding_box = AABB(min, max);
    return true;
}

void Triangle::Transform(Matrix3 transformation)
{
    for(auto & v : v_)
        v = transformation * v;

    auto normal_mat = transformation.Transposed();
    for(auto & v : n_)
        v = normal_mat * v;
}

bool Triangle::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    return TriangleMethods::Hit(ray, v_, n_, e_, t_min, t_max, record);
}
