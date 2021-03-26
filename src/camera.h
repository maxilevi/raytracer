/*
 * Created by Maximiliano Levi on 21/02/2021.
 */

#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H


#include <random>
#include "math/vector3.h"
#include "scene.h"

class Camera {
public:
    const int kAntialiasingSamples = 128;
    const double kGamma = 1.5;
    static const int kMaxLightBounces = 16;

    Camera(uint32_t width, uint32_t height) : width_(width), height_(height) {
        this->colors_ = std::unique_ptr<Vector3[]>(new Vector3[width * height]);
    };

    void Draw(Scene&);

    /* Accessors and mutators */
    inline uint32_t Width() const { return width_; }
    inline uint32_t Height() const { return height_; }
    inline Vector3 const * GetFrame() const { return this->colors_.get(); }

private:
    uint32_t width_;
    uint32_t height_;
    std::unique_ptr<Vector3[]> colors_;

    void NormalizeFrame();
    void ProcessRays(Scene* scene, const std::vector<std::pair<int, int>>& params);
};


#endif //RAYTRACER_CAMERA_H
