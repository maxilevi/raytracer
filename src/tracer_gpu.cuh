#ifndef RAYTRACER_TRACER_H
#define RAYTRACER_TRACER_H

#include "camera.h"
#include "scene.h"

void ProcessRays(const Camera& camera, const Scene& scene);

#endif