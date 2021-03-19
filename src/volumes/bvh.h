/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#ifndef RAYTRACER_BVH_H
#define RAYTRACER_BVH_H


#include "volume.h"
#include "../scene.h"

class Bvh : public Volume {
public:
    Bvh() = default;
    Bvh(std::vector<std::shared_ptr<Volume>>& volumes, size_t start, size_t end);

    bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const override;
    bool BoundingBox(AABB& bounding_box) const override;

private:
    AABB box_;
    std::shared_ptr<Volume> left_;
    std::shared_ptr<Volume> right_;
    size_t start_;
    size_t end_;
};


#endif //RAYTRACER_BVH_H
