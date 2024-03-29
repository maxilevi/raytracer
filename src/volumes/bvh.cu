/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#include <algorithm>
#include "bvh.h"
#include <random>

bool Bvh::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    /* If we don't hit the bounding box of the node then discard all of it's children. */
    if (!this->box_.Hit(ray, t_min, t_max)) {
        return false;
    }

    bool hit_left = this->left_->Hit(ray, t_min, t_max, record);
    bool hit_right = this->right_->Hit(ray, t_min, hit_left ? record.t : t_max, record);

    return hit_left || hit_right;
}

bool Bvh::BoundingBox(AABB &bounding_box) const
{
    bounding_box = this->box_;
    return true;
}

/* Constructs a BVH tree from a list of triangles */
Bvh::Bvh(std::vector<std::shared_ptr<Triangle>> &volumes, size_t start, size_t end)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 2);
    int axis = dist(gen);

    auto compare_lambda = [&](const std::shared_ptr<Volume> &a, const std::shared_ptr<Volume> &b) {
        AABB box_a;
        AABB box_b;

        if (!a->BoundingBox(box_a) || !b->BoundingBox(box_b))
            std::cerr << "No bounding box in bvh_node constructor." << std::endl;

        return box_a.Min()[axis] < box_b.Min()[axis];
    };
    /* We have 3 cases. Either we have 1, 2 or more triangles left. In the first two cases we set those as the leaf nodes.
     * In the third case we need to recursively subdivide the collection even more. Therefore, we sort the remaining triangles
     * according to a random axis and set the childs of the current node to be 2 new BVH trees generated from the remaining
     * elements.
     * */

    size_t objects_left = end - start;
    if (objects_left == 1) {
        left_ = right_ = volumes[start];
    } else if (objects_left == 2) {
        if (compare_lambda(volumes[start], volumes[start + 1])) {
            left_ = volumes[start];
            right_ = volumes[start + 1];
        } else {
            left_ = volumes[start + 1];
            right_ = volumes[start];
        }
    } else {
        std::sort(volumes.begin() + start, volumes.begin() + end, compare_lambda);
        size_t mid = start + objects_left / 2;
        left_ = std::make_shared<Bvh>(volumes, start, mid);
        right_ = std::make_shared<Bvh>(volumes, mid, end);
    }

    /* We create a bounding box for each node. This is key so that we will be able to discard half on each step */

    AABB box_left, box_right;

    if (!left_->BoundingBox(box_left) || !right_->BoundingBox(box_right))
        std::cerr << "No bounding box in bvh_node constructor." << std::endl;

    box_ = AABB::Merge(box_left, box_right);
    start_ = start;
    end_ = end;
}
