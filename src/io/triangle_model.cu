/*
 * Created by Maximiliano Levi on 03/03/2021.
 */

#include "triangle_model.h"
#include <assert.h>

TriangleModel::TriangleModel(std::unique_ptr<Triangle[]> triangles, uint32_t count)
{
    triangles_ = std::move(triangles);
    count_ = count;
};

void TriangleModel::Translate(Vector3 offset)
{
    for (uint32_t i = 0; i < count_; ++i)
    {
        triangles_[i].Translate(offset);
    }
}

void TriangleModel::Scale(Vector3 scale)
{
    for (uint32_t i = 0; i < count_; ++i)
    {
        triangles_[i].Scale(scale);
    }
}

void TriangleModel::Transform(Matrix3 transformation)
{
    for (uint32_t i = 0; i < count_; ++i)
    {
        triangles_[i].Transform(transformation);
    }
}

std::unique_ptr<TriangleModel> Copy(const TriangleModel* a)
{
    assert(a->triangles_ != nullptr);
    assert(a->Size() != 0);

    auto* copy = new Triangle[a->Size()];
    memcpy(copy, &a->triangles_[0], a->Size() * sizeof(Triangle));
    return std::make_unique<TriangleModel>(std::unique_ptr<Triangle[]>(copy), a->Size());
}

std::unique_ptr<TriangleModel> TriangleModel::Merge(const TriangleModel * a, const TriangleModel * b)
{
    if (a == nullptr)
        return Copy(b);

    if (b == nullptr)
        return Copy(a);

    size_t n = a->Size() + b->Size();
    auto* triangles = new Triangle[n];

    assert(a->triangles_ != nullptr);
    assert(a->Size() != 0);
    assert(b->triangles_ != nullptr);
    assert(b->Size() != 0);

    for (size_t i = 0; i < a->Size(); ++i)
        triangles[i] = a->triangles_[i];

    for (size_t i = 0; i < b->Size(); ++i)
        triangles[a->Size() + i] = b->triangles_[i];

    return std::make_unique<TriangleModel>(std::unique_ptr<Triangle[]>(triangles), (uint32_t)n);
}
