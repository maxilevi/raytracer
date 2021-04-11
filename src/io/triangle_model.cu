/*
 * Created by Maximiliano Levi on 03/03/2021.
 */

#include "triangle_model.h"

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
