#include "tracer_gpu.cuh"
#include "math/vector3.h"

/*
const double PI = 3.14159265;


Vector3 Camera::RandomPointOnUnitSphere(std::uniform_real_distribution<double> dist, std::mt19937 gen)
{
    double u1 = dist(gen);
    double u2 = dist(gen);
    double lambda = acos(2.0 * u1 - 1) - PI/2.0;
    double phi = 2.0 * PI * u2;
    return {std::cos(lambda) * std::cos(phi), std::cos(lambda) * std::sin(phi), std::sin(lambda)};
}

Vector3 BackgroundColor(const Ray& ray)
{
    auto unit_dir = Vector3(ray.Direction()).Normalized();
    double t = 0.5 * (unit_dir.Y() + 1.0);
    return (1.0 - t) * Vector3::One + t * Vector3(0.5, 0.7, 1.0);
}

Vector3 Camera::Color(const Scene& scene, const Ray& ray, std::uniform_real_distribution<double> dist, std::mt19937 gen) const
{
    Ray current_ray = ray;
    HitResult result;
    Vector3 color = Vector3::One;
    int iteration = 0;
    while (scene.Hit(current_ray, 0.001, std::numeric_limits<double>::max(), result))
    {
        Vector3 target_direction = result.Normal + RandomPointOnUnitSphere(dist, gen);
        current_ray = Ray(result.Point, target_direction);
        color *= 0.5;
        if (iteration++ == Camera::kMaxLightBounces)
            return {0, 0, 0};
    }
    return color * BackgroundColor(current_ray);
}
*/

void Trace(Scene& scene, const std::vector<std::pair<int, int>>& params, int n_colors, Vector3 origin, Vector3 screen, Vector3 step_x, Vector3 step_y, int width, int height)
{
    /*
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    int* d_params;
    cudaMalloc(&d_params, params.size() * sizeof(int) * 2);
    cudaMemcpy(d_params, &params[0], params.size() * sizeof(int) * 2, cudaMemcpyHostToDevice);

    double* out_colors;
    cudaMalloc(&out_colors, n_colors * sizeof(double));

    ProcessRays<<<(params.size() + 255) / 256, 256>>>(out_colors, d_params, n, origin, screen, step_x, step_y, width, height);

    // Copy back

    cudaFree(out_colors);
    cudaFree(d_params);*/
}


