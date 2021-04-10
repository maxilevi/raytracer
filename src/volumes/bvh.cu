/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#include <algorithm>
#include "bvh.h"
#include "../kernel/random.h"

CUDA_DEVICE bool Bvh::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    if (!this->box_.Hit(ray, t_min, t_max)) {

        return false;
    }

    bool hit_left = this->left_->Hit(ray, t_min, t_max, record);
    bool hit_right = this->right_->Hit(ray, t_min, hit_left ? record.t : t_max, record);

    return hit_left || hit_right;
}

CUDA_DEVICE bool Bvh::BoundingBox(AABB &bounding_box) const
{
    bounding_box = this->box_;
    return true;
}
template<class T, class F>
CUDA_DEVICE void Sort(T* start, T* end, F comparator)
{
    auto i = start + 1;
    while (i < end)
    {
        auto j = i;
        while(j > start && !comparator(*(j-1), *j))
        {
            auto tmp = j;
            *j = *(j-1);
            *(j-1) = *tmp;
            j--;
        }
        i++;
    }
}

CUDA_DEVICE Bvh::Bvh(Volume** volumes, size_t start, size_t end)
{
    this->is_left_leaf = true;
    this->is_right_leaf = true;
    this->volumes = volumes + start;
    this->volumes_size = end;

    uint32_t seed = start + end;
    int axis = RandomInt(seed, 0, 2);

    auto compare_lambda = [&](const Volume* a, const Volume* b)
    {
        AABB box_a;
        AABB box_b;

        //if (!a->BoundingBox(box_a) || !b->BoundingBox(box_b))
        //    std::cerr << "No bounding box in bvh_node constructor." << std::endl;

        return box_a.Min()[axis] < box_b.Min()[axis];
    };

    size_t objects_left = end - start;
    if (objects_left == 1)
    {
        left_ = right_ = volumes[start];
    }
    else if(objects_left == 2)
    {
        if (compare_lambda(volumes[start], volumes[start + 1])) {
            left_ = volumes[start];
            right_ = volumes[start + 1];
        } else {
            left_ = volumes[start + 1];
            right_ = volumes[start];
        }
    } else {
        Sort(volumes + start, volumes + end, compare_lambda);
        size_t mid = start + objects_left / 2;
        left_ = new Bvh(volumes, start, mid);
        right_ = new Bvh(volumes, mid, end);
        is_left_leaf = false;
        is_right_leaf = false;
    }

    AABB box_left, box_right;

    //if (!left_->BoundingBox (box_left) || !right_->BoundingBox(box_right))
    //    std::cerr << "No bounding box in bvh_node constructor." << std::endl;

    box_ = AABB::Merge(box_left, box_right);
    start_ = start;
    end_ = end;
}

CUDA_DEVICE Bvh::~Bvh()
{
    if(!is_left_leaf)
        delete (Bvh*) left_;
    if(!is_right_leaf)
        delete (Bvh*) right_;
}
