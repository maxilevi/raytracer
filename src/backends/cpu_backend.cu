/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#include "cpu_backend.h"
#include <thread>
#include <assert.h>

#define THREAD_COUNT 32

void CPUBackend::Trace(Scene &scene, const std::vector <std::pair<int, int>> &params, Vector3 *colors, Viewport& viewport)
{
    assert(scene.GetBvh() != nullptr);
    auto width = viewport.width;
    auto height = viewport.height;
    Vector3 origin = viewport.origin;
    Vector3 screen = viewport.view_port_lower_left_corner;
    Vector3 step_x = viewport.horizontal;
    Vector3 step_y = viewport.vertical;
    auto* bvh = scene.GetBvh();
    auto original_random = Random::New(false);

    auto element_func = [&](std::pair<int, int> pair)
    {
        auto i = pair.first;
        auto j = pair.second;
        auto seed = (i * width + j);
        auto random = original_random.Reseed(seed);
        double noise = random.Double();
        double u = (i + noise) / double(width);
        double v = (j + noise) / double(height);

        Ray r(origin, screen + step_x * u + step_y * v);
        return RenderingBackend::Color<Bvh>(bvh, r, random);
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