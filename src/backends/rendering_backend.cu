/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#include "rendering_backend.h"

CUDA_HOST_DEVICE Vector3 RenderingBackend::BackgroundColor(const Ray& ray)
{
    auto unit_dir = Vector3(ray.Direction()).Normalized();
    double t = 0.5 * (unit_dir.Y() + 1.0);
    return (1.0 - t) * Vector3(1) + t * Vector3(0.5, 0.7, 1.0);//(1.0 - t) * Vector3(double(180) / double(256)) + t * Vector3(double(124) / double(256));
}
