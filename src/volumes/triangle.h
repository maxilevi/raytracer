/*
 * Created by Maximiliano Levi on 25/02/2021.
 */

#ifndef RAYTRACER_TRIANGLE_H
#define RAYTRACER_TRIANGLE_H


#include "../math/vector3.h"
#include "../math/ray.h"
#include "volume.h"
#include "../math/matrix3.h"

class Triangle : public Volume {
public:
    Triangle() {};
    Triangle(Vector3 v0, Vector3 v1, Vector3 v2)
    {
        v_[0] = v0;
        v_[1] = v1;
        v_[2] = v2;

        auto normal = Vector3::Cross(v1 - v0, v2 - v0);
        n_[0] = normal;
        n_[1] = normal;
        n_[2] = normal;

        UpdateEdges();
    };

    Triangle(Vector3 v0, Vector3 v1, Vector3 v2, Vector3 n0, Vector3 n1, Vector3 n2)
    {
        v_[0] = v0;
        v_[1] = v1;
        v_[2] = v2;

        n_[0] = n0;
        n_[1] = n1;
        n_[2] = n2;

        UpdateEdges();
    };
    bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const override;
    bool BoundingBox(AABB& bounding_box) const override;
    void Translate(Vector3 offset);
    void Scale(Vector3 scale);
    void Transform(Matrix3 transformation);

    friend std::ostream& operator<<(std::ostream& stream, const Triangle& triangle);
    friend class GPUTriangle;

private:
    Vector3 v_[3];
    Vector3 n_[3];
    Vector3 e_[2];

    void UpdateEdges();
};


#endif //RAYTRACER_TRIANGLE_H
