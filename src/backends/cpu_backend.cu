/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#include "cpu_backend.h"
#include "../kernel/random.h"
#include <thread>
#include <random>
#define THREAD_COUNT 16


Vector3 Color(const Bvh* bvh, const Ray& ray, std::uniform_real_distribution<double> dist, std::mt19937 gen)
{
    Ray current_ray = ray;
    HitResult result;
    Vector3 color = Vector3(1);
    int iteration = 0;
    while (bvh->Hit(current_ray, 0.001, MAX_DOUBLE, result))
    {
        Vector3 target_direction = result.Normal + RenderingBackend::RandomPointOnUnitSphere(dist(gen), dist(gen));
        current_ray = Ray(result.Point, target_direction);
        color *= 0.5;
        if (iteration++ == RenderingBackend::kMaxLightBounces)
            return {0, 0, 0};
    }
    return color * RenderingBackend::BackgroundColor(current_ray);
}

void CPUBackend::Trace(Scene &scene, const std::vector <std::pair<int, int>> &params, Vector3 *colors, int width, int height)
{
    double screen_ratio = (double(width) / double(height));
    Vector3 origin(0, 0, 0);
    Vector3 screen(-screen_ratio, -1, -1);
    Vector3 step_x(std::abs(screen_ratio) * 2.0, 0, 0);
    Vector3 step_y(0, 2, 0);
    auto* bvh = scene.GetBvh();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto element_func = [&](std::pair<int, int> pair)
    {
        auto i = pair.first;
        auto j = pair.second;
        uint32_t seed = j * width + i;
        double noise = dist(gen);
        double u = (i + noise) / double(width);
        double v = (j + noise) / double(height);

        Ray r(origin, screen + step_x * u + step_y * v);
        colors[j * width + i] += Color(bvh, r, dist, gen);
    };

    auto slice_func = [&](size_t start, size_t end)
    {
        auto mn = end < params.size() ? end : params.size();
        for(size_t i = start; i < mn; ++i)
            element_func(params[i]);
    };

    std::thread threads[THREAD_COUNT];
    const size_t step = params.size() / THREAD_COUNT;
    std::cout << "Launching " << THREAD_COUNT << " threads with a step size of " << step << " each." << std::endl;
    for(size_t i = 0; i < THREAD_COUNT; ++i)
    {
        size_t offset = i * step;
        threads[i] = std::thread(slice_func, offset, offset + step);
    }

    for(auto & thread : threads)
    {
        thread.join();
    }
}