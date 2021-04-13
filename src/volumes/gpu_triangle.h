/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#ifndef RAYTRACER_GPU_TRIANGLE_H
#define RAYTRACER_GPU_TRIANGLE_H
#include "../math/vector3.h"
#include "hit_result.h"
#include "../math/ray.h"
#include "triangle.h"

class GPUTriangle {
public:
    GPUTriangle(Triangle* triangle)
    {
        for(int i = 0; i < 3; ++i)
        {
            v_[i] = triangle->v_[i];
            n_[i] = triangle->n_[i];
        }
    }
    CUDA_DEVICE bool Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const;

private:
    Vector3 v_[3];
    Vector3 n_[3];

    CUDA_DEVICE bool Intersects(const Ray &ray, double &t, double& u, double &v) const;
    CUDA_DEVICE bool Intersects2(const Ray &ray, double &t, double& u, double &v) const;
};


#endif //RAYTRACER_GPU_TRIANGLE_H
