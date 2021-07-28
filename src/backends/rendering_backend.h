/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#ifndef RAYTRACER_RENDERING_BACKEND_H
#define RAYTRACER_RENDERING_BACKEND_H
#include "../kernel/helper.h"
#include "../scenes/scene.h"
#include "../viewport.h"
#include "../kernel/random.h"
#include <vector>
#include <stdlib.h>

class RenderingBackend {
public:
    static const int kMaxLightBounces = 128;

    virtual void Trace(Scene& scene, const std::vector<std::pair<int, int>>& params, Vector3* colors, Viewport& viewport) = 0;
    static CUDA_HOST_DEVICE Vector3 BackgroundColor(const Ray& ray);


    template<class T>
    static CUDA_HOST_DEVICE Vector3 Color(const T* bvh, const Ray& ray, Random& random)
    {
        Ray current_ray = ray;
        HitResult result;
        double shade = 1.25;
        Vector3 color(1);
        bool any = false;
        int iteration = 0;
        while (bvh->Hit(current_ray, 0.001, MAX_DOUBLE, result))
        {
            if (!any)
            {
                any = true;
                color = result.Color;
            }

            if (iteration++ == RenderingBackend::kMaxLightBounces)
                return {0, 0, 0};

            shade *= 0.8;
            Ray scattered;
            Vector3 attenuation;
            if (result.Material->Scatter(current_ray, result, attenuation, scattered, random))
                color *= attenuation;

            current_ray = scattered;
        }
        if (any)
            return shade * color;
        return RenderingBackend::BackgroundColor(current_ray);
    }
};


#endif //RAYTRACER_RENDERING_BACKEND_H
