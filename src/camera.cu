/*
 * Created by Maximiliano Levi on 21/02/2021.
 */
#include "kernel/helper.h"
#include "camera.h"

void Camera::ProcessRays(Scene& scene, const std::vector<std::pair<int, int>>& params)
{
    Viewport view;
    view.height = height_;
    view.width = width_;
    view.origin = origin_;
    view.horizontal = horizontal_;
    view.vertical = vertical_;
    view.view_port_lower_left_corner = view_port_lower_left_corner_;
    this->backend_->Trace(scene, params, colors_.get(), view);
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

void Camera::Draw(Scene& scene)
{
    std::vector<std::pair<int, int>> params;
    for(int s = 0; s < Camera::kAntialiasingSamples; ++s) {
        for (int32_t j = height_ - 1; j > -1; --j) {
            for (int32_t i = 0; i < (int32_t) width_; ++i) {

                params.emplace_back(i, j);
                if (s == 0)
                    this->colors_[j * width_ + i] = Vector3();
            }
        }
    }

    this->ProcessRays(scene, params);

    this->NormalizeFrame();
}

void Camera::Configure(Vector3 position, Vector3 look_at, double fov)
{
    double screen_ratio = (double(width_) / double(height_));
    this->origin_ = Vector3(0, 0, 0);
    this->view_port_lower_left_corner_ = Vector3(-screen_ratio, -1, -1);
    this->horizontal_ = Vector3(std::abs(screen_ratio) * 2.0, 0, 0);
    this->vertical_ = Vector3(0, 2, 0);
    /*
     *     model->Translate(Vector3(-0.5, -5.5, -4.5));
+        //model->Transform(Matrix3::FromEuler({0, 0, 0}));
+        model->Translate(Vector3(-0.4, -6.5, -4.75));

         scene.Add(model);
    auto aspect_ratio = double(width_) / double(height_);
    auto theta = (fov * PI / 180.0);
    auto h = tan(theta / 2.0);

    auto viewport_height = 2.0 * h;
    auto viewport_width = aspect_ratio * viewport_height;

    auto w = (position - look_at).Normalized();
    auto u = Vector3::Cross(Vector3::UnitY, w).Normalized();
    auto v = Vector3::Cross(w, u);

    this->origin_ = position;
    this->horizontal_ = u * viewport_width;
    this->vertical_ = v * viewport_height;
    this->view_port_lower_left_corner_ = - horizontal_ / 2.0 - vertical_ / 2.0 - w;*/
}
