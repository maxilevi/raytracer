/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#ifndef RAYTRACER_RAY_H
#define RAYTRACER_RAY_H


#include "vector3.h"

class Ray {
public:
    Ray() = default;
    Ray(const Vector3& origin, const Vector3& direction) : origin_(origin), direction_(direction) {};
    [[nodiscard]] inline const Vector3& Origin() const { return this->origin_; }
    [[nodiscard]] inline const Vector3& Direction() const { return this->direction_; }
    [[nodiscard]] inline Vector3 Point(double t) const { return this->origin_ + this->direction_ * t; }

private:
    Vector3 origin_;
    Vector3 direction_;

};


#endif //RAYTRACER_RAY_H
