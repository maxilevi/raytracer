/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#ifndef RAYTRACER_RENDERING_BACKEND_H
#define RAYTRACER_RENDERING_BACKEND_H
#include "../kernel/helper.h"
#include "../scenes/scene.h"
#include <vector>

class RenderingBackend {
public:
    static const int kMaxLightBounces = 8;

    virtual void Trace(Scene& scene, const std::vector<std::pair<int, int>>& params, Vector3* colors, int width, int height) = 0;
    static CUDA_HOST_DEVICE Vector3 RandomPointOnUnitSphere(double u1, double u2);
    static CUDA_HOST_DEVICE Vector3 BackgroundColor(const Ray& ray);
};


#endif //RAYTRACER_RENDERING_BACKEND_H
