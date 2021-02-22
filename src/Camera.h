/*
 * Created by Maximiliano Levi on 21/02/2021.
 */

#ifndef RAYTRACER_CAMERA_H
#define RAYTRACER_CAMERA_H


#include "Vector3.h"

class Camera {
public:
    Camera(int width, int height) : width(width), height(height) {
        this->colors = std::unique_ptr<Vector3[]>(new Vector3[width * height]);
    };

    void Draw();
    void SetBackgroundColor(Vector3 color);
    void SetBackgroundGradient();
    Vector3 const * const GetFrame();


private:
    int width;
    int height;
    std::unique_ptr<Vector3[]> colors;
};


#endif //RAYTRACER_CAMERA_H
