/*
 * Created by Maximiliano Levi on 3/15/2021.
 */

#ifndef RAYTRACER_MATRIX3_H
#define RAYTRACER_MATRIX3_H


#include "vector3.h"

class Matrix3 {
public:
    Matrix3(Vector3 row0, Vector3 row1, Vector3 row2)
    {
        rows_[0] = row0;
        rows_[1] = row1;
        rows_[2] = row2;
    };

    static Matrix3 FromEuler(Vector3 angles);
    static Matrix3 FromRotationX(double angle);
    static Matrix3 FromRotationY(double angle);
    static Matrix3 FromRotationZ(double angle);

    friend Matrix3 operator *(const Matrix3& mat1, const Matrix3& mat2);

private:
    Vector3 rows_[3];
};


#endif //RAYTRACER_MATRIX3_H
