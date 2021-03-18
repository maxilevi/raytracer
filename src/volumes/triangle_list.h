/*
 * Created by Maximiliano Levi on 03/03/2021.
 */

#ifndef RAYTRACER_TRIANGLE_LIST_H
#define RAYTRACER_TRIANGLE_LIST_H


#include "volume.h"
#include "triangle.h"
#include "../math/matrix3.h"

class TriangleList : public Volume {
public:
    TriangleList(std::unique_ptr<Triangle[]> triangles, uint32_t count);

    bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const override;
    bool BoundingBox(AABB& bounding_box) const override;
    void Translate(Vector3 offset);
    void Scale(Vector3 scale);
    void Transform(Matrix3 transformation);

    [[nodiscard]] inline uint32_t Size() const { return count_; }

    std::unique_ptr<Triangle[]> triangles_;
private:
    uint32_t count_;
};


#endif //RAYTRACER_TRIANGLE_LIST_H
