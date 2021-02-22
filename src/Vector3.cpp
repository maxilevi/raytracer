/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#include "Vector3.h"

Vector3 &Vector3::operator-=(const Vector3 &V2)
{
    this->v[0] -= V2.X();
    this->v[1] -= V2.Y();
    this->v[2] -= V2.Z();
    return *this;
}

Vector3 &Vector3::operator+=(const Vector3 &V2)
{
    this->v[0] += V2.X();
    this->v[1] += V2.Y();
    this->v[2] += V2.Z();
    return *this;
}

Vector3 &Vector3::operator*=(const Vector3 &V2)
{
    this->v[0] *= V2.X();
    this->v[1] *= V2.Y();
    this->v[2] *= V2.Z();
    return *this;
}

Vector3 &Vector3::operator*=(const double Scalar)
{
    this->v[0] *= Scalar;
    this->v[1] *= Scalar;
    this->v[2] *= Scalar;
    return *this;
}

Vector3 &Vector3::operator/=(const Vector3 &V2)
{
    this->v[0] /= V2.X();
    this->v[1] /= V2.Y();
    this->v[2] /= V2.Z();
    return *this;
}

Vector3 &Vector3::operator/=(const double Scalar)
{
    this->v[0] /= Scalar;
    this->v[1] /= Scalar;
    this->v[2] /= Scalar;
    return *this;
}

Vector3 Vector3::operator +(const Vector3& V2)
{
    Vector3 temp(*this);
    temp += V2;
    return temp;
}

Vector3 Vector3::operator -(const Vector3& V2)
{
    Vector3 temp(*this);
    temp -= V2;
    return temp;
}

Vector3 Vector3::operator *(const Vector3& V2)
{
    Vector3 temp(*this);
    temp *= V2;
    return temp;
}

Vector3 Vector3::operator *(double Scalar)
{
    Vector3 temp(*this);
    temp *= Scalar;
    return temp;
}

Vector3 Vector3::operator /(const Vector3& V2)
{
    Vector3 temp(*this);
    temp /= V2;
    return temp;
}

Vector3 Vector3::operator /(double Scalar)
{
    Vector3 temp(*this);
    temp /= Scalar;
    return temp;
}

std::ostream& operator<<(std::ostream& os, const Vector3& Vector)
{
    os << "(" << Vector.X() << ", " << Vector.Y() << ", "  << Vector.Z() << ")";
    return os;
}

Vector3 operator*(const Vector3 Vector, double Scalar)
{
    return Vector * Scalar;
}

Vector3 operator+(const Vector3 V1, Vector3& V2)
{
    return V1 + V2;
}

const Vector3 Vector3::UnitX = Vector3(1, 0, 0);
const Vector3 Vector3::UnitY = Vector3(0, 1, 0);
const Vector3 Vector3::UnitZ = Vector3(0, 0, 1);
const Vector3 Vector3::Zero = Vector3(0, 0, 0);