/*
 * Created by Maximiliano Levi on 3/15/2021.
 */

#include "matrix3.h"
#include <cmath>

#define RADIANS(x)  (x * 3.14159265 / 180.0)

Matrix3 Matrix3::FromEuler(Vector3 angles)
{
    return FromRotationX(angles.X()) * FromRotationY(angles.Y()) * FromRotationZ(angles.Z());
}

Matrix3 Matrix3::FromRotationX(double angle)
{
    double theta = RADIANS(angle);
    return Matrix3(
        Vector3(1, 0, 0),
        Vector3(0, std::cos(theta), -std::sin(theta)),
        Vector3(0, std::sin(theta), std::cos(theta))
    );
}

Matrix3 Matrix3::FromRotationY(double angle)
{
    double theta = RADIANS(angle);
    return Matrix3(
        Vector3(std::cos(theta), 0, std::sin(theta)),
        Vector3(0, 1, 0),
        Vector3(-std::sin(theta), 0, std::cos(theta))
    );
}

Matrix3 Matrix3::FromRotationZ(double angle)
{
    double theta = RADIANS(angle);
    return Matrix3(
        Vector3(std::cos(theta),  -std::sin(theta), 0),
        Vector3(std::sin(theta), std::cos(theta), 0),
        Vector3(0, 0, 1)
    );
}

Matrix3 operator*(const Matrix3 &mat1, const Matrix3 &mat2)
{
    return Matrix3(
        Vector3(
                Vector3::Dot(mat1.rows_[0], mat2.Col0()),
                Vector3::Dot(mat1.rows_[1], mat2.Col0()),
                Vector3::Dot(mat1.rows_[2], mat2.Col0())
                ),
        Vector3(
                Vector3::Dot(mat1.rows_[0], mat2.Col0()),
                Vector3::Dot(mat1.rows_[1], mat2.Col1()),
                Vector3::Dot(mat1.rows_[2], mat2.Col1())
                ),
        Vector3(
                Vector3::Dot(mat1.rows_[0], mat2.Col2()),
                Vector3::Dot(mat1.rows_[1], mat2.Col2()),
                Vector3::Dot(mat1.rows_[2], mat2.Col2())
                )
        );
}

Vector3 operator*(const Matrix3 &mat1, const Vector3 &vec)
{
    return Vector3(
        Vector3::Dot(mat1.rows_[0], vec),
        Vector3::Dot(mat1.rows_[1], vec),
        Vector3::Dot(mat1.rows_[2], vec)
    );
}
