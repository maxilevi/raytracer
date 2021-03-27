/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#include "scene.h"
#include "volumes/triangle.h"

void Scene::Build(Triangle* device_volumes, size_t count)
{
    this->volumes_ = device_volumes;
    this->count_ = count;
}

CUDA_DEVICE bool Scene::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    HitResult temp;
    bool any_hit = false;
    double closest_so_far = t_max;
    for (size_t i = 0; i < count_; ++i)
    {
        if(this->volumes_[i].Hit(ray, t_min, closest_so_far, temp))
        {
            any_hit = true;
            closest_so_far = temp.t;
            record = temp;
        }
    }
    return any_hit;
}

CUDA_DEVICE bool Scene::BoundingBox(AABB &output_box) const
{
    if (count_ == 0) return false;

    AABB temp_box;
    bool first_box = true;

    for (size_t i = 0; i < count_; ++i)
    {
        const auto& object = volumes_[i];
        if (!object.BoundingBox(temp_box))
            return false;
        output_box = first_box ? temp_box : AABB::Merge(output_box, temp_box);
        first_box = false;
    }

    return true;
}

void Scene::Dispose()
{
    CUDA_CALL(cudaFree(volumes_));
}
