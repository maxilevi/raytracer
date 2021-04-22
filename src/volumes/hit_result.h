/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#ifndef RAYTRACER_HIT_RESULT_H
#define RAYTRACER_HIT_RESULT_H

class Material;

struct HitResult
{
    double u = 0, v = 0, t = 0;
    Vector3 Point;
    Vector3 Normal;
    Vector3 Color;
    const Material* Material;
};

#endif //RAYTRACER_HIT_RESULT_H
