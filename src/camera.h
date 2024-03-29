/*
 * Created by Maximiliano Levi on 21/02/2021.
 */

#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H


#include "math/vector3.h"
#include "scenes/scene.h"
#include "backends/rendering_backend.h"

class Camera {
public:
    const int kAntialiasingSamples = 1;
    const double kGamma = 1.5;

    Camera(uint32_t width, uint32_t height, std::unique_ptr<RenderingBackend>& backend) : width_(width), height_(height), backend_(std::move(backend))
    {
        this->colors_ = std::unique_ptr<Vector3[]>(new Vector3[width * height]);
    };
    void Configure(Vector3 position, Vector3 look_at, double fov);
    void Draw(Scene&);

    /* Accessors and mutators */
    inline uint32_t Width() const { return width_; }
    inline uint32_t Height() const { return height_; }
    inline Vector3 const * GetFrame() const { return this->colors_.get(); }

private:
    Vector3 view_port_lower_left_corner_;
    Vector3 origin_;
    Vector3 horizontal_;
    Vector3 vertical_;
    uint32_t width_;
    uint32_t height_;
    std::unique_ptr<RenderingBackend> backend_;
    std::unique_ptr<Vector3[]> colors_;

    void NormalizeFrame();
    void ProcessRays(Scene& scene, const std::vector<std::pair<int, int>>& params);
};


#endif //RAYTRACER_CAMERA_H
