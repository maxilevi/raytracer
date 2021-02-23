/*
 * Created by Maximiliano Levi on 21/02/2021.
 */

#include "Camera.h"
#include "Ray.h"

Vector3 Color(Ray& ray)
{
    auto unitDir = ray.Direction().Normalized();
    auto end = 0.5 * (ray.Direction().Normalized() + Vector3::One);
    return end;
}

void Camera::Draw()
{
    Vector3 origin(0, 0, 0);
    Vector3 screen(-2, -1, -1);
    Vector3 stepX(4, 0, 0);
    Vector3 stepY(0, 2, 0);

    for (uint32_t i = 0; i < width_; ++i)
    {
        for (uint32_t j = 0; j < height_; ++j)
        {
            double u = (double)i / width_;
            double v = (double)j / height_;
            Ray r(origin, screen + stepX * u + stepY * v);
            this->colors_[i * height_ + j] = Color(r);
        }
    }
}

void Camera::SetBackgroundColor(Vector3 color)
{

}