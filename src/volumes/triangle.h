/*
 * Created by Maximiliano Levi on 25/02/2021.
 */

#ifndef RAYTRACER_TRIANGLE_H
#define RAYTRACER_TRIANGLE_H


#include "../vector3.h"
#include "../ray.h"
#include "../volume.h"

class Triangle : public Volume {
public:
    Triangle(Vector3 v0, Vector3 v1, Vector3 v2)
    {
        v_[0] = v0;
        v_[1] = v1;
        v_[2] = v2;
    };
    bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const override;

private:
    bool Intersects(const Ray& ray, double t_min, double t_max, double& t) const;
    Vector3 v_[3];
};


#endif //RAYTRACER_TRIANGLE_H
