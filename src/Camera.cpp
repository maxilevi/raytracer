/*
 * Created by Maximiliano Levi on 21/02/2021.
 */

#include "Camera.h"
#include "Ray.h"

Vector3 Color(Ray& ray)
{
    return {1, 0, 1};
}

void Camera::Draw()
{
    Vector3 origin(0, 0, 0);
    Vector3 screenOrigin(-1, -1, -1);

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            double u = (double)i / width;
            double v = (double)j / height;
            Ray r(origin, screenOrigin + Vector3::UnitX * u + Vector3::UnitY * v);
            (*this->colors)[i * width + j] = Color(r);
        }
    }
}

void Camera::SetBackgroundColor(Vector3 color)
{

}

const Vector3 const *const Camera::GetFrame()
{
    return *this->colors;
}
