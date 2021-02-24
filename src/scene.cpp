/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#include "scene.h"

bool Scene::Add(Volume* volume)
{
    this->volumes_.push_back(volume);
    return false;
}

Volume *Scene::operator[](int idx)
{
    return this->volumes_[idx];
}

bool Scene::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    HitResult temp;
    bool any_hit = false;
    double closest_so_far = t_max;
    for (int i = 0; i < Size(); ++i)
    {
        if(this->volumes_[i]->Hit(ray, t_min, closest_so_far, temp))
        {
            any_hit = true;
            closest_so_far = temp.t;
            record = temp;
        }
    }
    return any_hit;
}
