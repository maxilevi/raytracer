#ifndef RAYTRACER_TRACER_H
#define RAYTRACER_TRACER_H

#include "camera.h"
#include "scene.h"

void Trace(Scene& scene, const std::vector<std::pair<int, int>>& params, int n_colors, Vector3 origin, Vector3 screen, Vector3 step_x, Vector3 step_y, int width, int height);

#endif