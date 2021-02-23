/*
 * Created by Maximiliano Levi on 21/02/2021.
 */

#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H


#include "Vector3.h"

class Camera {
public:
    Camera(uint32_t width, uint32_t height) : width(width), height(height) {
        this->colors = std::unique_ptr<Vector3[]>(new Vector3[width * height]);
    };

    void Draw();
    void SetBackgroundColor(Vector3 color);
    void SetBackgroundGradient();

    /* Accessors and mutators */
    inline uint32_t Width() const { return width; }
    inline uint32_t Height() const { return height; }
    inline Vector3 const * const GetFrame() const { return this->colors.get(); }

private:
    uint32_t width;
    uint32_t height;
    std::unique_ptr<Vector3[]> colors;
};


#endif //RAYTRACER_CAMERA_H
