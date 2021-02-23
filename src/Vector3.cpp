/*
 * Created by Maximiliano Levi on 16/02/2021.
 */

#include "Vector3.h"

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
    if (length)
        *this /= this->Length();
}
