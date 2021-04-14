/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#include "rendering_backend.h"
#include "../kernel/random.h"

CUDA_HOST_DEVICE Vector3 RenderingBackend::RandomPointOnUnitSphere(double u1, double u2)
{
    double lambda = acos(2.0 * u1 - 1) - PI / 2.0;
    double phi = 2.0 * PI * u2;
    return {std::cos(lambda) * std::cos(phi), std::cos(lambda) * std::sin(phi), std::sin(lambda)};
}

CUDA_HOST_DEVICE Vector3 RenderingBackend::BackgroundColor(const Ray& ray)
{
    auto unit_dir = Vector3(ray.Direction()).Normalized();
    double t = 0.5 * (unit_dir.Y() + 1.0);
    return (1.0 - t) * Vector3(1) + t * Vector3(0.5, 0.7, 1.0);
}