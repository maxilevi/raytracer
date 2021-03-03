/*
 * Created by Maximiliano Levi on 03/03/2021.
 */

#include "triangle_list.h"

TriangleList::TriangleList(std::unique_ptr<Triangle[]> triangles, uint32_t count)
{
    triangles_ = std::move(triangles);
    count_ = count;
};

bool TriangleList::Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const
{
    bool any_hit = false;
    double closest_so_far = t_max;
    for (uint32_t i = 0; i < count_; ++i)
    {
        if(triangles_[i].Hit(ray, t_min, closest_so_far, record))
        {
            any_hit = true;
            closest_so_far = record.t;
        }
    }
    return any_hit;
}

bool TriangleList::BoundingBox(AABB& bounding_box) const
{
    AABB aabb;

    if(!count_ || !triangles_[0].BoundingBox(aabb))
        return false;

    for(uint32_t i = 1; i < count_; ++i)
    {
        AABB tri_aabb;
        if(!triangles_[i].BoundingBox(tri_aabb))
            return false;
        aabb = AABB::Merge(aabb, tri_aabb);
    }
    return true;
}

void TriangleList::Translate(Vector3 offset)
{
    for (uint32_t i = 0; i < count_; ++i)
    {
        triangles_[i].Translate(offset);
    }
}

void TriangleList::Scale(Vector3 scale)
{
    for (uint32_t i = 0; i < count_; ++i)
    {
        triangles_[i].Scale(scale);
    }
}