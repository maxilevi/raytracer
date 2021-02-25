/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#ifndef RAYTRACER_VECTOR3_H
#define RAYTRACER_VECTOR3_H

#include <cmath>
#include <iostream>

class Vector3 {
public:
    Vector3(): Vector3(0, 0, 0) {};

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
    Vector3& operator *=(const double scalar);
    Vector3& operator /=(const Vector3& vector);
    Vector3& operator /=(double Scalar);

    Vector3 operator +(const Vector3& vector);
    Vector3 operator -(const Vector3& vector);
    Vector3 operator *(const Vector3& vector);
    Vector3 operator /(const Vector3& vector);

    inline double operator[](int idx) const { return v_[idx]; }

    friend std::ostream& operator<<(std::ostream& stream, const Vector3& vector);
    friend Vector3 operator +(const Vector3 v1, Vector3& v2);
    friend Vector3 operator *(const Vector3 vector, const double scalar);
    friend Vector3 operator *(const double scalar, const Vector3 vector);
    friend Vector3 operator /(const Vector3 vector, const double scalar);

    static double Dot(const Vector3& a, const Vector3& b);

    static const Vector3 UnitX;
    static const Vector3 UnitY;
    static const Vector3 UnitZ;
    static const Vector3 Zero;
    static const Vector3 One;


private:
    double v_[3];
};

#endif //RAYTRACER_VECTOR3_H
