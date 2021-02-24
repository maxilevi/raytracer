/*
 * Created by Maximiliano Levi on 21/02/2021.
 */

#include "Camera.h"
#include "Ray.h"
#include <limits>

Vector3 Color(const Scene& scene, const Ray& ray)
{
    HitResult result;
    if (scene.Hit(ray, 0.0, std::numeric_limits<double>::max(), result))
    {
        return 0.5 * (result.Normal + Vector3::One);
    }
    else
    {
        auto unit_dir = ray.Direction().Normalized();
        double t = 0.5 * (unit_dir.Y() + 1.0);
        return (1.0 - t) * Vector3::One + t * Vector3(0.5, 0.7, 1.0);
    }
}

void Camera::Draw(Scene& scene)
{
    Vector3 origin(0, 0, 0);
    Vector3 screen(-2, -1, -1);
    Vector3 step_x(4, 0, 0);
    Vector3 step_y(0, 2, 0);

    for (int32_t i = 0; i < width_; ++i)
    {
        for (int32_t j = height_-1; j > -1; --j)
        {
            double u = double(i) / double(width_);
            double v = double(j) / double(height_);
            Ray r(origin, screen + step_x * u + step_y * v);
            this->colors_[j * width_ + i] = Color(scene, r);
        }
    }
}

void Camera::SetBackgroundColor(Vector3 color)
{

}