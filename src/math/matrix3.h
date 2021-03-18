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

    [[nodiscard]] inline Vector3 Col(int i) const { return Vector3(rows_[0][i], rows_[1][i], rows_[2][i]); }
    [[nodiscard]] inline Vector3 Col0() const { return Col(0); }
    [[nodiscard]] inline Vector3 Col1() const { return Col(1); }
    [[nodiscard]] inline Vector3 Col2() const { return Col(2); }

    [[nodiscard]] static Matrix3 FromEuler(Vector3 angles);
    [[nodiscard]] static Matrix3 FromRotationX(double angle);
    [[nodiscard]] static Matrix3 FromRotationY(double angle);
    [[nodiscard]] static Matrix3 FromRotationZ(double angle);

    friend Matrix3 operator *(const Matrix3& mat1, const Matrix3& mat2);
    friend Vector3 operator *(const Matrix3& mat1, const Vector3& vec);
    inline Vector3& operator[](int idx) { return rows_[idx]; }

private:
    Vector3 rows_[3];
};


#endif //RAYTRACER_MATRIX3_H
