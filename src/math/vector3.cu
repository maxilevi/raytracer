/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#include "vector3.h"

const Vector3 Vector3::UnitX = Vector3(1, 0, 0);
const Vector3 Vector3::UnitY = Vector3(0, 1, 0);
const Vector3 Vector3::UnitZ = Vector3(0, 0, 1);
const Vector3 Vector3::Zero = Vector3(0, 0, 0);
const Vector3 Vector3::One = Vector3(1, 1, 1);

CUDA_HOST_DEVICE Vector3 Vector3::Normalized()
{
    Vector3 tmp(*this);
    tmp.Normalize();
    return tmp;
}

CUDA_HOST_DEVICE void Vector3::Normalize()
{
    auto length = this->Length();
    if (length > 0.0001)
        *this /= length;
}

CUDA_HOST_DEVICE double Vector3::Dot(const Vector3 &a, const Vector3 &b)
{
    return a.v_[0] * b.v_[0] + a.v_[1] * b.v_[1] + a.v_[2] * b.v_[2];
}

CUDA_HOST_DEVICE Vector3 Vector3::Cross(const Vector3 &a, const Vector3 &b)
{
    return {
        a.v_[1] * b.v_[2] - a.v_[2] * b.v_[1],
        -(a.v_[0] * b.v_[2] - a.v_[2] * b.v_[0]),
        a.v_[0] * b.v_[1] - a.v_[1] * b.v_[0]
    };
}

CUDA_HOST_DEVICE Vector3& Vector3::operator +=(const Vector3& vector)
{
    this->v_[0] += vector.v_[0];
    this->v_[1] += vector.v_[1];
    this->v_[2] += vector.v_[2];
    return *this;
}

CUDA_HOST_DEVICE Vector3& Vector3::operator -=(const Vector3& vector)
{
    this->v_[0] -= vector.v_[0];
    this->v_[1] -= vector.v_[1];
    this->v_[2] -= vector.v_[2];
    return *this;
}

CUDA_HOST_DEVICE Vector3& Vector3::operator *=(const Vector3& vector)
{
    this->v_[0] *= vector.v_[0];
    this->v_[1] *= vector.v_[1];
    this->v_[2] *= vector.v_[2];
    return *this;
}

CUDA_HOST_DEVICE Vector3& Vector3::operator *=(const double& scalar)
{
    this->v_[0] *= scalar;
    this->v_[1] *= scalar;
    this->v_[2] *= scalar;
    return *this;
}

CUDA_HOST_DEVICE Vector3& Vector3::operator /=(const Vector3& vector)
{
    this->v_[0] /= vector.v_[0];
    this->v_[1] /= vector.v_[1];
    this->v_[2] /= vector.v_[2];
    return *this;
}

CUDA_HOST_DEVICE Vector3& Vector3::operator /=(const double& Scalar)
{
    this->v_[0] /= Scalar;
    this->v_[1] /= Scalar;
    this->v_[2] /= Scalar;
    return *this;
}

CUDA_HOST_DEVICE Vector3 Vector3::Lerp(const Vector3 &a, const Vector3 &b, double t)
{
    return (1.0 - t) * a + b * t;
}

std::ostream& operator<<(std::ostream& stream, const Vector3& vector)
{
    stream << "(" << vector.v_[0] << ", " << vector.v_[1] << ", "  << vector.v_[2] << ")";
    return stream;
}
