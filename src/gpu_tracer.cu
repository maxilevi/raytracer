/*
 * Created by Maximiliano Levi on 3/19/2021.
 */

#include "gpu_tracer.h"
#include "defines.h"
#include "math/ray.h"
#include "camera.h"

__constant__ const double PI = 3.14159265;
__constant__ const double MAX_DOUBLE = DBL_MAX;

CUDA_DEVICE double RandomDouble()
{
    return 0;
}

CUDA_DEVICE Vector3 RandomPointOnUnitSphere()
{
    double u1 = RandomDouble();
    double u2 = RandomDouble();
    double lambda = acos(2.0 * u1 - 1) - PI/2.0;
    double phi = 2.0 * PI * u2;
    return {std::cos(lambda) * std::cos(phi), std::cos(lambda) * std::sin(phi), std::sin(lambda)};
}

CUDA_DEVICE Vector3 BackgroundColor(const Ray& ray)
{
    auto unit_dir = Vector3(ray.Direction()).Normalized();
    double t = 0.5 * (unit_dir.Y() + 1.0);
    return (1.0 - t) * Vector3(1) + t * Vector3(0.5, 0.7, 1.0);
}

CUDA_DEVICE Vector3 Color(const Scene& scene, const Ray& ray)
{
    Ray current_ray = ray;
    HitResult result;
    Vector3 color = Vector3(1);
    int iteration = 0;
    while (scene.Hit(current_ray, 0.001, std::numeric_limits<double>::max(), result))
    {
        Vector3 target_direction = result.Normal + RandomPointOnUnitSphere();
        current_ray = Ray(result.Point, target_direction);
        color *= 0.5;
        if (iteration++ == Camera::kMaxLightBounces)
            return {0, 0, 0};
    }
    return color * BackgroundColor(current_ray);
}

__global__
void CUDAColor(Scene scene, Vector3* out_colors, const int* device_params, int n, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int i = device_params[2*idx];
    int j = device_params[2*idx+1];
    out_colors[j * width + i] += Vector3(1, 0, 0);
}

void GPUTrace(Scene scene, const std::vector<std::pair<int, int>>& params, Vector3* colors, int width, int height)
{
    double screen_ratio = (double(width) / double(height));
    Vector3 origin(0, 0, 0);
    Vector3 screen(-screen_ratio, -1, -1);
    Vector3 step_x(std::abs(screen_ratio) * 2.0, 0, 0);
    Vector3 step_y(0, 2, 0);

    int n = params.size();
    int blocks = (n+255)/256;
    int threads = 256;

    std::cout << "Launching " << blocks << " blocks with " << threads << " threads each (" << (blocks * threads) << " total threads)" << std::endl;

    Vector3* out_colors;
    int* device_params;

    CUDA_CALL(cudaMalloc(&device_params, n * 2 * sizeof(int)));
    CUDA_CALL(cudaMalloc(&out_colors, width * height * sizeof(Vector3)));

    CUDA_CALL(cudaMemcpy(device_params, &params[0], n * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(out_colors, colors, width * height * sizeof(Vector3), cudaMemcpyHostToDevice));

    std::cout << "Launched." << std::endl;

    CUDAColor<<<blocks, threads>>>(scene, out_colors, device_params, n, width, height);

    cudaDeviceSynchronize();

    std::cout << "All CUDA threads joined." << std::endl;

    CUDA_CALL(cudaMemcpy(colors, out_colors, width * height * sizeof(Vector3), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(out_colors));
    CUDA_CALL(cudaFree(device_params));
}