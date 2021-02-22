/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#ifndef RAYTRACER_RAY_H
#define RAYTRACER_RAY_H


#include "Vector3.h"

class Ray {
public:
    Ray();
    Ray(const Vector3& origin, const Vector3& direction) : origin(origin), direction(direction) {};
    inline Vector3 Origin() const { return this->origin; }
    inline Vector3 Direction() const { return this->direction; }
    inline Vector3 Point(float t) const { return this->origin + this->direction * t; }

private:
    Vector3 origin;
    Vector3 direction;

};


#endif //RAYTRACER_RAY_H
