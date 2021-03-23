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
    CUDA_DEVICE Vector3(): Vector3(0, 0, 0) {};

    CUDA_DEVICE explicit Vector3(double scale): Vector3(scale, scale, scale) {};

    CUDA_DEVICE  Vector3(double x, double y, double z) {
        v_[0] = x;
        v_[1] = y;
        v_[2] = z;
    };

    CUDA_DEVICE Vector3 Normalized();
    CUDA_DEVICE void Normalize();
    CUDA_DEVICE inline double LengthSquared() const { return (v_[0] * v_[0] + v_[1] * v_[1] + v_[2] * v_[2]); }
    CUDA_DEVICE inline double Length() const { return std::sqrt(LengthSquared()); }

    CUDA_DEVICE inline double X() const { return v_[0]; }
    CUDA_DEVICE inline double Y() const { return v_[1]; }
    CUDA_DEVICE inline double Z() const { return v_[2]; }

    CUDA_DEVICE inline Vector3& operator +() { return *this; }
    CUDA_DEVICE inline Vector3 operator -() { return {-v_[0], -v_[1], -v_[2]}; }

    CUDA_DEVICE Vector3& operator +=(const Vector3& vector);
    CUDA_DEVICE Vector3& operator -=(const Vector3& vector);
    CUDA_DEVICE Vector3& operator *=(const Vector3& vector);
    CUDA_DEVICE Vector3& operator *=(const double& scalar);
    CUDA_DEVICE Vector3& operator /=(const Vector3& vector);
    CUDA_DEVICE Vector3& operator /=(const double& Scalar);

    CUDA_DEVICE inline double& operator[](int idx) { return v_[idx]; }
    CUDA_DEVICE inline const double& operator[](int idx) const { return v_[idx]; }

    friend std::ostream& operator<<(std::ostream& stream, const Vector3& vector);

    CUDA_DEVICE inline friend Vector3 operator +(const Vector3& v1, const Vector3& v2)
    {
        return {v1.v_[0] + v2.v_[0], v1.v_[1] + v2.v_[1], v1.v_[2] + v2.v_[2]};
    }

    CUDA_DEVICE inline friend Vector3 operator *(const Vector3& v1, const Vector3& v2)
    {
        return {v1.v_[0] * v2.v_[0], v1.v_[1] * v2.v_[1], v1.v_[2] * v2.v_[2]};
    }

    CUDA_DEVICE inline friend Vector3 operator -(const Vector3& v1, const Vector3& v2)
    {
        return {v1.v_[0] - v2.v_[0], v1.v_[1] - v2.v_[1], v1.v_[2] - v2.v_[2]};
    }

    CUDA_DEVICE inline friend Vector3 operator *(const Vector3& vector, const double& scalar)
    {
        return {vector.v_[0] * scalar, vector.v_[1] * scalar, vector.v_[2] * scalar};
    }

    CUDA_DEVICE inline friend Vector3 operator *(const double& scalar, const Vector3& vector)
    {
        return vector * scalar;
    }

    CUDA_DEVICE inline friend Vector3 operator /(const Vector3& vector, const double& scalar)
    {
        return {vector.v_[0] / scalar, vector.v_[1] / scalar, vector.v_[2] / scalar};
    }

    CUDA_DEVICE static double Dot(const Vector3& a, const Vector3& b);
    CUDA_DEVICE static Vector3 Cross(const Vector3& a, const Vector3& b);

    static const Vector3 UnitX;
    static const Vector3 UnitY;
    static const Vector3 UnitZ;
    static const Vector3 Zero;
    static const Vector3 One;

private:
    double v_[3]{};
};

#endif //RAYTRACER_VECTOR3_H
