/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#include "bvh.h"

bool Bvh::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    if (!this->box_.Hit(ray, t_min, t_max))
        return false;

    bool hit_left = this->left_->Hit(ray, t_min, t_max, record);
    bool hit_right = this->right_->Hit(ray, t_min, hit_left ? record.t : t_max, record);

    return hit_left || hit_right;
}

bool Bvh::BoundingBox(AABB &bounding_box) const
{
    bounding_box = this->box_;
    return true;
}

Bvh::Bvh(const Scene &scene)
{

}
