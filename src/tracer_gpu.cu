#include "tracer_gpu.cuh"
#include "math/vector3.h"

__global__
void DoProcess(float* out_colors)
{

}

void ProcessRays(const Camera& camera, const Scene& scene)
{
    float* out_colors;
    cudaMallocManaged(&out_colors, camera.Width() * camera.Height() * sizeof(Vector3));

    //DoProcess<<<1, 1>>> (out_colors);

    cudaDeviceSynchronize();
    cudaFree(out_colors);
}