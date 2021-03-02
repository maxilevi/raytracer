/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#include "vector3.h"

const Vector3 Vector3::UnitX = Vector3(1, 0, 0);
const Vector3 Vector3::UnitY = Vector3(0, 1, 0);
const Vector3 Vector3::UnitZ = Vector3(0, 0, 1);
const Vector3 Vector3::Zero = Vector3(0, 0, 0);
const Vector3 Vector3::One = Vector3(1, 1, 1);

Vector3 Vector3::Normalized()
{
    Vector3 tmp(*this);
    tmp.Normalize();
    return tmp;
}

void Vector3::Normalize()
{
    auto length = this->Length();
    if (length > 0.0001)
        *this /= length;
}

double Vector3::Dot(const Vector3 &a, const Vector3 &b)
{
    return a.v_[0] * b.v_[0] + a.v_[1] * b.v_[1] + a.v_[2] * b.v_[2];
}

Vector3 Vector3::Cross(const Vector3 &a, const Vector3 &b)
{
    return {
        a.v_[1] * b.v_[2] - a.v_[2] * b.v_[1],
        -(a.v_[0] * b.v_[2] - a.v_[2] * b.v_[0]),
        a.v_[0] * b.v_[1] - a.v_[1] * b.v_[0]
    };
}

Vector3& Vector3::operator +=(const Vector3& vector)
{
    this->v_[0] += vector.v_[0];
    this->v_[1] += vector.v_[1];
    this->v_[2] += vector.v_[2];
    return *this;
}

Vector3& Vector3::operator -=(const Vector3& vector)
{
    this->v_[0] -= vector.v_[0];
    this->v_[1] -= vector.v_[1];
    this->v_[2] -= vector.v_[2];
    return *this;
}

Vector3& Vector3::operator *=(const Vector3& vector)
{
    this->v_[0] *= vector.v_[0];
    this->v_[1] *= vector.v_[1];
    this->v_[2] *= vector.v_[2];
    return *this;
}

Vector3& Vector3::operator *=(const double scalar)
{
    this->v_[0] *= scalar;
    this->v_[1] *= scalar;
    this->v_[2] *= scalar;
    return *this;
}

Vector3& Vector3::operator /=(const Vector3& vector)
{
    this->v_[0] /= vector.v_[0];
    this->v_[1] /= vector.v_[1];
    this->v_[2] /= vector.v_[2];
    return *this;
}

Vector3& Vector3::operator /=(double Scalar)
{
    this->v_[0] /= Scalar;
    this->v_[1] /= Scalar;
    this->v_[2] /= Scalar;
    return *this;
}

Vector3 Vector3::operator +(const Vector3& vector)
{
    Vector3 temp(*this);
    temp += vector;
    return temp;
}

Vector3 Vector3::operator *(const Vector3& vector)
{
    Vector3 temp(*this);
    temp *= vector;
    return temp;
}

Vector3 Vector3::operator /(const Vector3& vector)
{
    Vector3 temp(*this);
    temp /= vector;
    return temp;
}

std::ostream& operator<<(std::ostream& stream, const Vector3& vector)
{
    stream << "(" << vector.v_[0] << ", " << vector.v_[1] << ", "  << vector.v_[2] << ")";
    return stream;
}

Vector3 operator +(const Vector3 v1, Vector3& v2)
{
    return {v1.v_[0] + v2.v_[0], v1.v_[1] + v2.v_[1], v1.v_[2] + v2.v_[2]};
}

Vector3 operator *(const Vector3 vector, const double scalar)
{
    return {vector.v_[0] * scalar, vector.v_[1] * scalar, vector.v_[2] * scalar};
}

Vector3 operator *(const double scalar, const Vector3 vector)
{
    return vector * scalar;
}

Vector3 operator /(const Vector3 vector, const double scalar)
{
    return {vector.v_[0] / scalar, vector.v_[1] / scalar, vector.v_[2] / scalar};
}

Vector3 operator-(Vector3 v1, Vector3 v2)
{
    return {v1.v_[0] - v2.v_[0], v1.v_[1] - v2.v_[1], v1.v_[2] - v2.v_[2]};
}