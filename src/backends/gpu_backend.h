/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#ifndef RAYTRACER_GPU_BACKEND_H
#define RAYTRACER_GPU_BACKEND_H
#include <vector>
#include "rendering_backend.h"

class GPUBackend : public RenderingBackend {
public:
    void Trace(Scene& scene, const std::vector<std::pair<int, int>>& params, Vector3* colors, int width, int height) override;
};


#endif //RAYTRACER_GPU_BACKEND_H
