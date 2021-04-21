/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#ifndef RAYTRACER_RENDERING_BACKEND_H
#define RAYTRACER_RENDERING_BACKEND_H
#include "../kernel/helper.h"
#include "../scenes/scene.h"
#include "../viewport.h"
#include <vector>
#include <stdlib.h>
#include <random>

class RenderingBackend {
public:
    static const int kMaxLightBounces = 128;
    static const int kRandomCount = 65536;

    virtual void Trace(Scene& scene, const std::vector<std::pair<int, int>>& params, Vector3* colors, Viewport& viewport) = 0;
    static CUDA_HOST_DEVICE Vector3 RandomPointOnUnitSphere(double u1, double u2);
    static CUDA_HOST_DEVICE Vector3 BackgroundColor(const Ray& ray);

    static double* GetRandomArray()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        auto random_numbers = new double[kRandomCount];
        for(int i = 0; i < kRandomCount; ++i)
        {
            random_numbers[i] = dist(gen);
        }
        return random_numbers;
    }

    static CUDA_HOST_DEVICE double RandomDouble(const double* random_numbers, uint32_t& seed)
    {
        auto val = random_numbers[seed % kRandomCount];
        seed = (seed * 1337) % kRandomCount;
        return val;
    }

    template<class T>
    static CUDA_HOST_DEVICE Vector3 Color(const T* bvh, const Ray& ray, const double* randoms, uint32_t& seed)
    {
        //return Vector3(1) * RandomDouble(randoms, seed);
        Ray current_ray = ray;
        HitResult result;
        double shade = 1.25;
        Vector3 color;
        bool any = false;
        int iteration = 0;
        while (bvh->Hit(current_ray, 0.001, MAX_DOUBLE, result))
        {
            if (!any)
            {
                any = true;
                color = result.Color;
            }
            Vector3 target_direction = result.Normal + RenderingBackend::RandomPointOnUnitSphere(RandomDouble(randoms, seed), RandomDouble(randoms, seed));
            current_ray = Ray(result.Point, target_direction);
            shade *= 0.8;

            if (iteration++ == RenderingBackend::kMaxLightBounces)
                return {0, 0, 0};
        }
        if (any)
            return shade * color;
        return RenderingBackend::BackgroundColor(current_ray);
    }
};


#endif //RAYTRACER_RENDERING_BACKEND_H
