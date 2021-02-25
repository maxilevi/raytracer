/*
 * Created by Maximiliano Levi on 21/02/2021.
 */

#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H


#include <random>
#include "Vector3.h"
#include "scene.h"

class Camera {
public:
    const int kAntialiasingSamples = 16;
    const double kGamma = 1.5;
    const int kMaxLightBounces = 8;

    Camera(uint32_t width, uint32_t height) : width_(width), height_(height) {
        this->colors_ = std::unique_ptr<Vector3[]>(new Vector3[width * height]);
    };

    void Draw(Scene&);
    void SetBackgroundColor(Vector3 color);
    void SetBackgroundGradient();

    /* Accessors and mutators */
    inline uint32_t Width() const { return width_; }
    inline uint32_t Height() const { return height_; }
    inline Vector3 const * const GetFrame() const { return this->colors_.get(); }

private:
    uint32_t width_;
    uint32_t height_;
    std::unique_ptr<Vector3[]> colors_;

    Vector3 RandomPointOnUnitSphere(std::uniform_real_distribution<double> dist, std::mt19937 gen);
    Vector3 Color(const Scene& scene, const Ray& ray, std::uniform_real_distribution<double> dist, std::mt19937 gen, int iteration = 0);
};


#endif //RAYTRACER_CAMERA_H
