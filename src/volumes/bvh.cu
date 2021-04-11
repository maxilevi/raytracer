/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#include <algorithm>
#include "bvh.h"
#include "../kernel/random.h"
#include <thrust/sort.h>

#include <stdio.h>
#include <assert.h>

CUDA_DEVICE bool Bvh::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    if (!this->box_.Hit(ray, t_min, t_max))
    {
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

CUDA_DEVICE void do_print(Volume** volumes, size_t index)
{
    //printf("Accessing %d\n", (int)index);
    printf("%p\n", volumes[index]);
}

template<class F>
CUDA_DEVICE void Sort(Volume** volumes, int start, int end, F comparator)
{
    printf("sorting!!!\n");
    auto i = start + 1;
    while (i < end)
    {
        int j = i;
        while(j > start && !comparator(volumes[j-1], volumes[j]))
        {
            auto tmp = volumes[j];
            volumes[j] = volumes[(j-1)];
            volumes[(j-1)] = tmp;
            j--;
        }
        i++;
    }
    printf("sorted.\n");
}

CUDA_DEVICE Bvh::Bvh(Volume** volumes, size_t start, size_t end)
{
    printf("Creating BVH wth Start: %d, End: %d, Volumes: %p\n", (int)start, (int)end, volumes);
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

        if (!a->BoundingBox(box_a) || !b->BoundingBox(box_b))
            printf("No bounding box in bvh_node constructor.");

        return box_a.Min()[axis] < box_b.Min()[axis];
    };

    size_t objects_left = end - start;
    printf("Left: %d\n", (int)objects_left);
    if (objects_left == 1)
    {
        do_print(volumes, start);
        left_ = right_ = volumes[start];
    }
    else if(objects_left == 2)
    {
        do_print(volumes, start);
        do_print(volumes, start+1);
        if (compare_lambda(volumes[start], volumes[start + 1])) {
            left_ = volumes[start];
            right_ = volumes[start + 1];
        } else {
            left_ = volumes[start + 1];
            right_ = volumes[start];
        }
    } else if (objects_left > 2)
    {
        Sort(volumes, start, end, compare_lambda);
        size_t mid = start + objects_left / 2;
        printf("Cutting in half. Start: %d, End: %d, Mid: %d\n", (int)start, (int)end, (int) mid);
        left_ = new Bvh(volumes, start, mid);
        right_ = new Bvh(volumes, mid, end);
        is_left_leaf = false;
        is_right_leaf = false;
    }

    AABB box_left, box_right;
    if (!left_->BoundingBox(box_left) || !right_->BoundingBox(box_right))
        printf("No bounding box in bvh_node constructor.");

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

void Bvh::BuildBvhRanges(std::vector<BvhRange>& ranges, std::shared_ptr<TriangleModel>)
{
    return BuildBvhRangesAux();
}

int Bvh::BuildBvhRangesAux(std::vector<BvhRange>& ranges, const Triangle* triangles, int s, int e)
{
    if (e - s > 2)
    {
        int mid = (s + e) / 2;
        int left_idx = BuildBvhRangesAux(ranges, triangles, s, mid);
        int right_idx = BuildBvhRangesAux(ranges,triangles,  mid, e);
        ranges.push_back({s, e, left_idx, right_idx});
        return ranges.size() - 1;
    }
    else if (e - s == 2)
    {
        ranges.push_back({s, e, left_idx, right_idx});
    }
    else
    {

    }
}

