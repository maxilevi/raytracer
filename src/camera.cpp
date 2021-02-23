/*
 * Created by Maximiliano Levi on 21/02/2021.
 */

#include "Camera.h"
#include "Ray.h"

Vector3 Color(Ray& ray)
{
    auto end = ray.Direction() * 0.5 + Vector3::One * 0.5;
    return end.Normalized();
}

void Camera::Draw()
{
    Vector3 origin(0, 0, 0);
    Vector3 screenOrigin(-1, -1, -1);

    for (uint32_t i = 0; i < width; ++i)
    {
        for (uint32_t j = 0; j < height; ++j)
        {
            double u = (double)i / width;
            double v = (double)j / height;
            Ray r(origin, screenOrigin + Vector3::UnitX * u * 2 + Vector3::UnitY * v * 2);
            this->colors[i * height + j] = Color(r);
        }
    }
}

void Camera::SetBackgroundColor(Vector3 color)
{

}