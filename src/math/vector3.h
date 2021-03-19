/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#ifndef RAYTRACER_VECTOR3_H
#define RAYTRACER_VECTOR3_H

#include <cmath>
#include <iostream>
#include "../defines.h"

class Vector3 {
public:
    Vector3(): Vector3(0, 0, 0) {};

    explicit Vector3(double scale): Vector3(scale, scale, scale) {};

    Vector3(double x, double y, double z) {
        v_[0] = x;
        v_[1] = y;
        v_[2] = z;
    };

    Vector3 Normalized();
    void Normalize();
    inline double LengthSquared() const { return (v_[0] * v_[0] + v_[1] * v_[1] + v_[2] * v_[2]); }
    inline double Length() const { return std::sqrt(LengthSquared()); }

    inline double X() const { return v_[0]; }
    inline double Y() const { return v_[1]; }
    inline double Z() const { return v_[2]; }

    inline Vector3& operator +() { return *this; }
    inline Vector3 operator -() { return {-v_[0], -v_[1], -v_[2]}; }

    Vector3& operator +=(const Vector3& vector);
    Vector3& operator -=(const Vector3& vector);
    Vector3& operator *=(const Vector3& vector);
    Vector3& operator *=(const double& scalar);
    Vector3& operator /=(const Vector3& vector);
    Vector3& operator /=(const double& Scalar);

    inline double& operator[](int idx) { return v_[idx]; }
    inline const double& operator[](int idx) const { return v_[idx]; }

    friend std::ostream& operator<<(std::ostream& stream, const Vector3& vector);

    inline friend Vector3 operator +(const Vector3& v1, const Vector3& v2)
    {
        return {v1.v_[0] + v2.v_[0], v1.v_[1] + v2.v_[1], v1.v_[2] + v2.v_[2]};
    }

    inline friend Vector3 operator *(const Vector3& v1, const Vector3& v2)
    {
        return {v1.v_[0] * v2.v_[0], v1.v_[1] * v2.v_[1], v1.v_[2] * v2.v_[2]};
    }

    inline friend Vector3 operator -(const Vector3& v1, const Vector3& v2)
    {
        return {v1.v_[0] - v2.v_[0], v1.v_[1] - v2.v_[1], v1.v_[2] - v2.v_[2]};
    }

    inline friend Vector3 operator *(const Vector3& vector, const double& scalar)
    {
        return {vector.v_[0] * scalar, vector.v_[1] * scalar, vector.v_[2] * scalar};
    }

    inline friend Vector3 operator *(const double& scalar, const Vector3& vector)
    {
        return vector * scalar;
    }

    inline friend Vector3 operator /(const Vector3& vector, const double& scalar)
    {
        return {vector.v_[0] / scalar, vector.v_[1] / scalar, vector.v_[2] / scalar};
    }

    static double Dot(const Vector3& a, const Vector3& b);
    static Vector3 Cross(const Vector3& a, const Vector3& b);

    static const Vector3 UnitX;
    static const Vector3 UnitY;
    static const Vector3 UnitZ;
    static const Vector3 Zero;
    static const Vector3 One;


private:
    double v_[3]{};
};

#endif //RAYTRACER_VECTOR3_H
