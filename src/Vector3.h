/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#ifndef RAYTRACER_VECTOR3_H
#define RAYTRACER_VECTOR3_H

#include <cmath>
#include <iostream>

class Vector3 {
public:
    Vector3() {};

    Vector3(double x, double y, double z) {
        v[0] = x;
        v[1] = y;
        v[2] = z;
    };

    inline double LengthSquared() const { return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
    inline double Length() const { return std::sqrt(LengthSquared()); }

    inline double X() const { return v[0]; }
    inline double Y() const { return v[1]; }
    inline double Z() const { return v[2]; }

    inline Vector3& operator +() { return *this; }
    inline Vector3 operator -() { return {-v[0], -v[1], -v[2]}; }

    inline Vector3& operator +=(const Vector3&);
    inline Vector3& operator -=(const Vector3&);
    inline Vector3& operator *=(const Vector3&);
    inline Vector3& operator *=(double);
    inline Vector3& operator /=(const Vector3&);
    inline Vector3& operator /=(double Scalar);

    inline Vector3 operator +(const Vector3&);
    inline Vector3 operator -(const Vector3&);
    inline Vector3 operator *(const Vector3&);
    inline Vector3 operator *(double);
    inline Vector3 operator /(const Vector3&);
    inline Vector3 operator /(double);

    inline friend Vector3 operator +(Vector3, Vector3&);
    inline friend Vector3 operator *(Vector3, double);
    inline friend std::ostream& operator<<(std::ostream&, const Vector3&);

    static const Vector3 UnitX;
    static const Vector3 UnitY;
    static const Vector3 UnitZ;
    static const Vector3 Zero;


private:
    double v[3];
};


#endif //RAYTRACER_VECTOR3_H
