/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#ifndef RAYTRACER_BVH_H
#define RAYTRACER_BVH_H


#include "volume.h"
#include "../scene.h"
#include <vector>

class Bvh : public Volume {
public:
    Bvh() = default;
    CUDA_DEVICE ~Bvh();
    CUDA_DEVICE Bvh(Volume** volumes, size_t start, size_t end);

    CUDA_DEVICE bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const override;
    CUDA_DEVICE bool BoundingBox(AABB& bounding_box) const override;

    Volume** volumes;
    size_t volumes_size;
private:
    AABB box_;
    Volume* left_;
    Volume* right_;
    size_t start_;
    size_t end_;
    bool is_left_leaf;
    bool is_right_leaf;
};


#endif //RAYTRACER_BVH_H
