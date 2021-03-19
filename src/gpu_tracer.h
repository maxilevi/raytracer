/*
 * Created by Maximiliano Levi on 3/19/2021.
 */

#ifndef RAYTRACER_GPU_TRACER_H
#define RAYTRACER_GPU_TRACER_H

#include "scene.h"
#include <vector>
#include "math/vector3.h"

void GPUTrace(Scene& scene, const std::vector<std::pair<int, int>>& params, Vector3* colors, int width, int height);

#endif //RAYTRACER_GPU_TRACER_H
