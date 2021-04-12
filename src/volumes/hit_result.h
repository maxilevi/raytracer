/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#ifndef RAYTRACER_HIT_RESULT_H
#define RAYTRACER_HIT_RESULT_H

struct HitResult
{
    double t = 0;
    Vector3 Point;
    Vector3 Normal;
};

#endif //RAYTRACER_HIT_RESULT_H
