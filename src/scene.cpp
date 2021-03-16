/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#include "scene.h"

bool Scene::Add(std::shared_ptr<Volume> volume)
{
    this->volumes_.push_back(volume);
    return false;
}

bool Scene::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    HitResult temp;
    bool any_hit = false;
    double closest_so_far = t_max;
    auto* volumes = &this->volumes_.front();
    const size_t size = this->volumes_.size();
    for (size_t i = 0; i < size; ++i)
    {
        if(volumes[i]->Hit(ray, t_min, closest_so_far, temp))
        {
            any_hit = true;
            closest_so_far = temp.t;
            record = temp;
        }
    }
    return any_hit;
}

bool Scene::BoundingBox(AABB &output_box) const
{
    if (volumes_.empty()) return false;

    AABB temp_box;
    bool first_box = true;

    for (const auto& object : volumes_)
    {
        if (!object->BoundingBox(temp_box))
            return false;
        output_box = first_box ? temp_box : AABB::Merge(output_box, temp_box);
        first_box = false;
    }

    return true;
}
