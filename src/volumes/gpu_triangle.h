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
        e_[0] = v_[1] - v_[0];
        e_[1] = v_[2] - v_[0];
    }
    CUDA_DEVICE bool Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const;

private:
    Vector3 v_[3];
    Vector3 n_[3];
    Vector3 e_[2];
};


#endif //RAYTRACER_GPU_TRIANGLE_H