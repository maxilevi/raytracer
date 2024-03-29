/*
 * Created by Maximiliano Levi on 25/02/2021.
 */

#include "triangle.h"
#include "triangle_methods.h"
#include <limits>
#include <algorithm>
#include <assert.h>

std::ostream& operator<<(std::ostream& stream, const Triangle& triangle)
{
    stream << "(" << triangle.v_[0] << ", " << triangle.v_[1] << ", "  << triangle.v_[2] << ", ";
    stream << triangle.n_[0] << ", " << triangle.n_[1] << ", "  << triangle.n_[2] << ",";
    stream << triangle.t_[0] << ", " << triangle.t_[1] << ")";
    return stream;
}

void Triangle::Translate(Vector3 offset)
{
    for(auto & i : v_)
        i += offset;
    UpdateEdges();
}

void Triangle::Scale(Vector3 scale)
{
    for(auto & i : v_)
        i *= scale;
    UpdateEdges();
}

bool Triangle::BoundingBox(AABB &bounding_box) const
{
    Vector3 min(MAX_DOUBLE), max(MIN_DOUBLE);
    for(auto& v : v_)
    {
        for(int i = 0; i < 3; ++i)
        {
            min[i] = MIN(v[i], min[i]);
            max[i] = MAX(v[i], max[i]);
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

    UpdateEdges();
}

bool Triangle::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    return TriangleMethods::Hit(ray, v_, n_, e_, t_, material_.get(), t_min, t_max, record);
}

void Triangle::UpdateEdges()
{
    e_[0] = v_[1] - v_[0];
    e_[1] = v_[2] - v_[0];
}
