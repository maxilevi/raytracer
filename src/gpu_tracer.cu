/*
 * Created by Maximiliano Levi on 3/19/2021.
 */

#include "gpu_tracer.h"
#include "kernel/helper.h"
#include "math/ray.h"
#include "camera.h"
#include "volumes/bvh.h"
#include "kernel/random.h"
#include "volumes/gpu_bvh.h"

#define THREAD_COUNT 256

CUDA_DEVICE Vector3 RandomPointOnUnitSphere(uint32_t& seed)
{
    double u1 = RandomDouble(seed);
    double u2 = RandomDouble(seed);
    double lambda = acos(2.0 * u1 - 1) - PI / 2.0;
    double phi = 2.0 * PI * u2;
    return {std::cos(lambda) * std::cos(phi), std::cos(lambda) * std::sin(phi), std::sin(lambda)};
}

CUDA_DEVICE Vector3 BackgroundColor(const Ray& ray)
{
    auto unit_dir = Vector3(ray.Direction()).Normalized();
    double t = 0.5 * (unit_dir.Y() + 1.0);
    return (1.0 - t) * Vector3(1) + t * Vector3(0.5, 0.7, 1.0);
}

CUDA_DEVICE Vector3 Color(GPUBvh& bvh, const Ray& ray, uint32_t& seed)
{
    Ray current_ray = ray;
    HitResult result;
    Vector3 color = Vector3(1);
    int iteration = 0;
    while (bvh.Hit(current_ray, 0.001, MAX_DOUBLE, result))
    {
        Vector3 target_direction = result.Normal + RandomPointOnUnitSphere(seed);
        current_ray = Ray(result.Point, target_direction);
        color *= 0.5;
        if (iteration++ == Camera::kMaxLightBounces)
            return {0, 0, 0};
    }
    return color * BackgroundColor(current_ray);
}

__global__
void ColorKernel(GPUBvh bvh, Vector3* out_colors, const int* device_params, int n, int width, int height, Vector3 origin, Vector3 screen, Vector3 step_x, Vector3 step_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t seed = idx;
    int i = device_params[2*idx];
    int j = device_params[2*idx+1];
    double noise = RandomDouble(seed);
    double u = (i + noise) / double(width);
    double v = (j + noise) / double(height);

    Ray r(origin, screen + step_x * u + step_y * v);
    out_colors[idx] = Color(bvh, r, seed);
}

void GPUTrace(Scene& scene, const std::vector<std::pair<int, int>>& params, Vector3* colors, int width, int height)
{
    double screen_ratio = (double(width) / double(height));
    Vector3 origin(0, 0, 0);
    Vector3 screen(-screen_ratio, -1, -1);
    Vector3 step_x(std::abs(screen_ratio) * 2.0, 0, 0);
    Vector3 step_y(0, 2, 0);

    int n = params.size();
    int step = n / 2;

    std::cout << "Total samples to process = " << n << std::endl;

    Vector3* out_colors;
    int* device_params;
    auto* all_samples = new Vector3[step];

    CUDA_CALL(cudaMalloc(&device_params, step * 2 * sizeof(int)));
    CUDA_CALL(cudaMalloc(&out_colors, step * sizeof(Vector3)));

    std::cout << "Generating GPU Bvh" << std::endl;

    GPUBvh gpu_bvh = GPUBvh::FromBvh(scene.GetBvh());

    std::cout << "Starting CUDA work" << std::endl;

    for(size_t w = 0; w < n; w += step)
    {
        std::cout << "Processing elements (" << w << ", " << w + step << ")" << std::endl;
        int size = MIN(step, params.size() - w);
        int blocks = (size + THREAD_COUNT - 1) / THREAD_COUNT;
        std::cout << "Launching " << blocks << " blocks with " << THREAD_COUNT << " threads each (" << (blocks * THREAD_COUNT) << " total threads) for " << step << " elements" << std::endl;

        CUDA_CALL(cudaMemcpy(device_params, &params[0] + w, size * 2 * sizeof(int), cudaMemcpyHostToDevice));

        ColorKernel<<<blocks, THREAD_COUNT>>>(gpu_bvh, out_colors, device_params, size, width, height, origin, screen, step_x, step_y);

        CUDA_CALL(cudaPeekAtLastError())
        CUDA_CALL(cudaDeviceSynchronize());

        std::cout << "All CUDA threads joined." << std::endl;

        std::cout << "Copying CUDA results." << std::endl;

        CUDA_CALL(cudaMemcpy(all_samples, out_colors, size * sizeof(Vector3), cudaMemcpyDeviceToHost));

        std::cout << "Collapsing samples..." << std::endl;

        /* Reduce all samples into one */
        for (size_t k = 0; k < size; ++k) {
            int i = params[k + w].first;
            int j = params[k + w].second;
            auto color = all_samples[k];
            colors[j * width + i] += color;
        }
    }

    delete[] all_samples;
    CUDA_CALL(cudaFree(out_colors));
    CUDA_CALL(cudaFree(device_params));
    GPUBvh::Delete(gpu_bvh);

    std::cout << "Finished CUDA work." << std::endl;
}