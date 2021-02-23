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

    Vector3 Normalized();
    void Normalize();
    inline double LengthSquared() const { return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
    inline double Length() const { return std::sqrt(LengthSquared()); }

    inline double X() const { return v[0]; }
    inline double Y() const { return v[1]; }
    inline double Z() const { return v[2]; }

    inline Vector3& operator +() { return *this; }
    inline Vector3 operator -() { return {-v[0], -v[1], -v[2]}; }

    inline Vector3& operator +=(const Vector3& vector)
    {
        this->v[0] -= vector.v[0];
        this->v[1] -= vector.v[1];
        this->v[2] -= vector.v[2];
        return *this;
    }

    inline Vector3& operator -=(const Vector3& vector)
    {
        this->v[0] += vector.v[0];
        this->v[1] += vector.v[1];
        this->v[2] += vector.v[2];
        return *this;
    }

    inline Vector3& operator *=(const Vector3& vector)
    {
        this->v[0] *= vector.v[0];
        this->v[1] *= vector.v[1];
        this->v[2] *= vector.v[2];
        return *this;
    }

    inline Vector3& operator *=(const double scalar)
    {
        this->v[0] *= scalar;
        this->v[1] *= scalar;
        this->v[2] *= scalar;
        return *this;
    }

    inline Vector3& operator /=(const Vector3& vector)
    {
        this->v[0] /= vector.v[0];
        this->v[1] /= vector.v[1];
        this->v[2] /= vector.v[2];
        return *this;
    }

    inline Vector3& operator /=(double Scalar)
    {
        this->v[0] /= Scalar;
        this->v[1] /= Scalar;
        this->v[2] /= Scalar;
        return *this;
    }

    inline Vector3 operator +(const Vector3& vector)
    {
        Vector3 temp(*this);
        temp += vector;
        return temp;
    }

    inline Vector3 operator -(const Vector3& vector)
    {
        Vector3 temp(*this);
        temp -= vector;
        return temp;
    }

    inline Vector3 operator *(const Vector3& vector)
    {
        Vector3 temp(*this);
        temp *= vector;
        return temp;
    }


    inline Vector3 operator /(const Vector3& vector)
    {
        Vector3 temp(*this);
        temp /= vector;
        return temp;
    }

    inline double operator[](int idx) const
    {
        return v[idx];
    }

    inline friend std::ostream& operator<<(std::ostream& stream, const Vector3& vector)
    {
        stream << "(" << vector.v[0] << ", " << vector.v[1] << ", "  << vector.v[2] << ")";
        return stream;
    }

    inline friend Vector3 operator +(const Vector3 v1, Vector3& v2)
    {
        return v1 + v2;
    }

    inline friend Vector3 operator *(const Vector3 vector, const double scalar)
    {
        return {vector.v[0] * scalar, vector.v[1] * scalar, vector.v[2] * scalar};
    }

    inline friend Vector3 operator /(const Vector3 vector, const double scalar)
    {
        return {vector.v[0] / scalar, vector.v[1] / scalar, vector.v[2] / scalar};
    }



    static const Vector3 UnitX;
    static const Vector3 UnitY;
    static const Vector3 UnitZ;
    static const Vector3 Zero;
    static const Vector3 One;


private:
    double v[3];
};

#endif //RAYTRACER_VECTOR3_H
