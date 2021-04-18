/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#include "cpu_backend.h"
#include "../kernel/random.h"
#include <thread>
#include <random>
#include <assert.h>

#define THREAD_COUNT 32


Vector3 Color(const Bvh* bvh, const Ray& ray, std::uniform_real_distribution<double> dist, std::mt19937 gen)
{
    Ray current_ray = ray;
    HitResult result;
    Vector3 color = Vector3(1);
    int iteration = 0;
    bool any = false;
    Vector3 first_color;
    while (bvh->Hit(current_ray, 0.001, MAX_DOUBLE, result))
    {
        if (!any) {
            first_color = result.Color;
            any = true;
        }
        Vector3 target_direction = result.Normal + RenderingBackend::RandomPointOnUnitSphere(dist(gen), dist(gen));
        current_ray = Ray(result.Point, target_direction);
        color *= 0.8;
        if (iteration++ == RenderingBackend::kMaxLightBounces)
            return {0, 0, 0};
    }
    if (any)
        return first_color;
    return color * RenderingBackend::BackgroundColor(current_ray);
}

void CPUBackend::Trace(Scene &scene, const std::vector <std::pair<int, int>> &params, Vector3 *colors, int width, int height)
{
    assert(scene.GetBvh() != nullptr);

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
        double noise = dist(gen);
        double u = (i + noise) / double(width);
        double v = (j + noise) / double(height);

        Ray r(origin, screen + step_x * u + step_y * v);
        return Color(bvh, r, dist, gen);
    };

    auto slice_func = [&](size_t start, size_t end, std::shared_ptr<Vector3[]> buffer)
    {
        auto mn = end < params.size() ? end : params.size();
        int k = 0;
        for(size_t i = start; i < mn; ++i)
            buffer[k++] += element_func(params[i]);
    };

    std::thread threads[THREAD_COUNT];
    std::shared_ptr<Vector3[]> buffers[THREAD_COUNT];
    const size_t step = params.size() / THREAD_COUNT;
    std::cout << "Launching " << THREAD_COUNT << " threads with a step size of " << step << " each." << std::endl;
    for(size_t i = 0; i < THREAD_COUNT; ++i)
    {
        size_t offset = i * step;
        buffers[i] = std::shared_ptr<Vector3[]>(new Vector3[step]);
        threads[i] = std::thread(slice_func, offset, offset + step, buffers[i]);
    }

    size_t offset = 0;
    for(size_t i = 0; i < THREAD_COUNT; ++i)
    {
        threads[i].join();
        for(size_t j = 0; j < step; ++j)
        {
            auto pair = params[offset + j];
            auto x = pair.first, y = pair.second;
            colors[y * width + x] += buffers[i][j];
        }
        offset = (offset + step) % (width * height);
    }
}