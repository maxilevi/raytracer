/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#ifndef RAYTRACER_SCENE_H
#define RAYTRACER_SCENE_H


#include "volumes/triangle.h"
#include "kernel/kernel_vector.h"
#include "kernel/kernel_ptr.h"

class Scene {
public:
    void Dispose();
    void Build(Triangle*, size_t);
    CUDA_DEVICE bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const;
    CUDA_DEVICE bool BoundingBox(AABB& output_box) const;

private:
    Triangle* volumes_;
    size_t count_;
};


#endif //RAYTRACER_SCENE_H
