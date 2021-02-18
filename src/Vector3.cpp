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

Vector3 &Vector3::operator*=(const float Scalar)
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

Vector3 &Vector3::operator/=(const float Scalar)
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

Vector3 Vector3::operator *(float Scalar)
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

Vector3 Vector3::operator /(float Scalar)
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