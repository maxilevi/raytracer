/*
 * Created by Maximiliano Levi on 21/02/2021.
 */
#include "kernel/helper.h"
#include "camera.h"
#include "gpu_tracer.h"

void Camera::ProcessRays(Scene& scene, const std::vector<std::pair<int, int>>& params)
{
    GPUTrace(scene, params, colors_.get(), width_, height_);
}

void Camera::NormalizeFrame()
{
    for (int32_t j = (int32_t)height_-1; j > -1; --j)
    {
        for (int32_t i = 0; i < (int32_t)width_; ++i)
        {
            auto color = this->colors_[j * width_ + i];

            /* Normalize the samples for antialiasing */
            //std::cout << color << std::endl;
            color /= Camera::kAntialiasingSamples;

            /* Gamma correction */
            color = Vector3(std::pow(color.X(), 1.0/kGamma), std::pow(color.Y(), 1.0 / kGamma), std::pow(color.Z(), 1.0 / kGamma));

            this->colors_[j * width_ + i] = color;
        }
    }
}

void Camera::Draw(Scene& scene)
{
    std::vector<std::pair<int, int>> params;
    for (int32_t j = height_-1; j > -1; --j)
    {
        for (int32_t i = 0; i < (int32_t)width_; ++i)
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