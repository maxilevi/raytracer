/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#ifndef RAYTRACER_BVH_H
#define RAYTRACER_BVH_H


#include "volume.h"
#include <vector>
#include "triangle.h"

class Bvh : public Volume {
public:
    Bvh() = default;
    Bvh(std::vector<std::shared_ptr<Triangle>>& volumes, size_t start, size_t end);
    bool BoundingBox(AABB& bounding_box) const override;

private:
    AABB box_;
    std::shared_ptr<Volume> left_;
    std::shared_ptr<Volume> right_;
    size_t start_;
    size_t end_;

    friend class GPUBvh;
    friend class GPUBvhNode;
};


#endif //RAYTRACER_BVH_H
