/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#ifndef RAYTRACER_CPU_BACKEND_H
#define RAYTRACER_CPU_BACKEND_H
#include "rendering_backend.h"

class CPUBackend : public RenderingBackend {
public:
    void Trace(Scene& scene, const std::vector<std::pair<int, int>>& params, Vector3* colors, Viewport& viewport) override;
};


#endif //RAYTRACER_CPU_BACKEND_H
