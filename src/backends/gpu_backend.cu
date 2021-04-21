/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#include "gpu_backend.h"
#include "../kernel/helper.h"
#include "../math/ray.h"
#include "../volumes/bvh.h"
#include "../volumes/gpu_bvh.h"
#include <chrono>
#include "../helper.h"
#include "../kernel/random.h"

#define THREAD_COUNT 512

__global__
void ColorKernel(GPUBvh bvh, double* randoms, Vector3* out_colors, const int* device_params, int n, int width, int height, Vector3 origin, Vector3 screen, Vector3 step_x, Vector3 step_y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t seed = idx * 2047;
    int i = device_params[2*idx];
    int j = device_params[2*idx+1];
    double noise = RenderingBackend::RandomDouble(randoms, seed);
    double u = (i + noise) / double(width);
    double v = (j + noise) / double(height);

    Ray r(origin, screen + step_x * u + step_y * v);
    out_colors[idx] = RenderingBackend::Color<GPUBvh>(&bvh, r, randoms, seed);
}

void GPUBackend::Trace(Scene &scene, const std::vector<std::pair<int, int>>& params, Vector3 *colors, Viewport& viewport)
{
    auto width = viewport.width;
    auto height = viewport.height;
    Vector3 origin = viewport.origin;
    Vector3 screen = viewport.view_port_lower_left_corner;
    Vector3 step_x = viewport.horizontal;
    Vector3 step_y = viewport.vertical;
    auto cpu_randoms = std::unique_ptr<double>(GetRandomArray());

    size_t n = params.size();
    size_t step = n / ((int)ceil(n / (float)33177600));

    std::cout << "Total samples to process = " << n << std::endl;

    double* gpu_randoms;
    CUDA_CALL(cudaMalloc(&gpu_randoms, sizeof(double) * RenderingBackend::kRandomCount));
    CUDA_CALL(cudaMemcpy(gpu_randoms, cpu_randoms.get(), sizeof(double) * RenderingBackend::kRandomCount, cudaMemcpyHostToDevice));

    Vector3* out_colors;
    int* device_params;
    auto all_samples = std::unique_ptr<Vector3>(new Vector3[step]);

    CUDA_CALL(cudaMalloc(&device_params, step * 2 * sizeof(int)));
    CUDA_CALL(cudaMalloc(&out_colors, step * sizeof(Vector3)));

    std::cout << "Generating GPU Bvh" << std::endl;

    auto gpu_result = GPUBvh::FromBvh(scene.GetBvh());
    auto gpu_bvh = gpu_result.first;
    auto gpu_materials = gpu_result.second;

    std::cout << "Starting CUDA work" << std::endl;


    for(size_t w = 0; w < n; w += step)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Processing elements (" << w << ", " << w + step << ")" << std::endl;
        size_t size = MIN(step, params.size() - w);
        size_t blocks = (size + THREAD_COUNT - 1) / THREAD_COUNT;
        std::cout << "Launching " << blocks << " blocks with " << THREAD_COUNT << " threads each (" << (blocks * THREAD_COUNT) << " total threads) for " << step << " elements" << std::endl;

        CUDA_CALL(cudaMemcpy(device_params, &params[0] + w, size * 2 * sizeof(int), cudaMemcpyHostToDevice));

        ColorKernel<<<blocks, THREAD_COUNT>>>(gpu_bvh, gpu_randoms, out_colors, device_params, size, width, height, origin, screen, step_x, step_y);

        CUDA_CALL(cudaPeekAtLastError())
        CUDA_CALL(cudaDeviceSynchronize());

        std::cout << "All CUDA threads joined, took " << TimeIt(t1) << " ms" << std::endl;

        std::cout << "Copying CUDA results." << std::endl;

        CUDA_CALL(cudaMemcpy(all_samples.get(), out_colors, size * sizeof(Vector3), cudaMemcpyDeviceToHost));

        std::cout << "Collapsing samples..." << std::endl;

        /* Reduce all samples into one */
        for (size_t k = 0; k < size; ++k) {
            int i = params[k + w].first;
            int j = params[k + w].second;
            auto color = all_samples.get()[k];
            colors[j * width + i] += color;
        }
    }

    CUDA_CALL(cudaFree(out_colors));
    CUDA_CALL(cudaFree(device_params));
    CUDA_CALL(cudaFree(gpu_randoms));
    GPUBvh::Delete(gpu_bvh);
    for(size_t i = 0; i < gpu_materials.size(); ++i)
    {
        gpu_materials[i].FreeGPUMaterial();
    }

    std::cout << "Finished CUDA work." << std::endl;
}