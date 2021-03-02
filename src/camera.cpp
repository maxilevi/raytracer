/*
 * Created by Maximiliano Levi on 21/02/2021.
 */
#include "defines.h"
#include "camera.h"
#include "ray.h"
#include <limits>
#if USE_CUDA
#include "tracer_gpu.cuh"
#else
#include <execution>
#endif


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
    auto unit_dir = ray.Direction().Normalized();
    double t = 0.5 * (unit_dir.Y() + 1.0);
    return (1.0 - t) * Vector3::One + t * Vector3(0.5, 0.7, 1.0);
}

Vector3 Camera::Color(const Scene& scene, const Ray& ray, std::uniform_real_distribution<double> dist, std::mt19937 gen, int iteration)
{
    if (iteration == Camera::kMaxLightBounces)
        return {0, 0, 0};

    HitResult result;
    if (scene.Hit(ray, 0.01, std::numeric_limits<double>::max(), result))
    {
        Vector3 target = result.Point + result.Normal + RandomPointOnUnitSphere(dist, gen);
        return 0.5 * Color(scene, Ray(result.Point, target - result.Point), dist, gen, iteration + 1);
    }
    else
    {
        return BackgroundColor(ray);
    }
}

void Camera::NormalizeFrame()
{
    for (int32_t j = (int32_t)height_-1; j > -1; --j)
    {
        for (int32_t i = 0; i < (int32_t)width_; ++i)
        {
            auto color = this->colors_[j * width_ + i];

            /* Normalize the samples for antialiasing */
            color /= Camera::kAntialiasingSamples;

            /* Gamma correction */
            color = Vector3(std::pow(color.X(), 1.0/kGamma), std::pow(color.Y(), 1.0 / kGamma), std::pow(color.Z(), 1.0 / kGamma));

            this->colors_[j * width_ + i] = color;
        }
    }
}

void Camera::ProcessRays(Scene& scene, std::vector<std::pair<int, int>> params)
{
    double screen_ratio = (double(width_) / double(height_));
    Vector3 origin(0, 0, 0);
    Vector3 screen(-screen_ratio, -1, -1);
    Vector3 step_x(std::abs(screen_ratio) * 2.0, 0, 0);
    Vector3 step_y(0, 2, 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::for_each(std::execution::par_unseq, params.begin(), params.end(), [&](std::pair<int, int> pair)
    {
        auto [i, j] = pair;
        double noise = dist(gen);
        double u = (i + noise) / double(width_);
        double v = (j + noise) / double(height_);

        Ray r(origin, screen + step_x * u + step_y * v);
        this->colors_[j * width_ + i] += Color(scene, r, dist, gen);
    });
}

void Camera::Draw(Scene& scene)
{
    std::vector<std::pair<int, int>> params;

    for (int32_t j = height_-1; j > -1; --j)
    {
        for (int32_t i = 0; i < width_; ++i)
        {
            for(int s = 0; s < Camera::kAntialiasingSamples; ++s)
            {
                params.emplace_back(i, j);
            }
            this->colors_[j * width_ + i] = Vector3();
        }
    }

    this->ProcessRays(scene, params);

    this->NormalizeFrame();
}

void Camera::SetBackgroundColor(Vector3 color)
{

}