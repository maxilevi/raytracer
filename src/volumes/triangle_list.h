/*
 * Created by Maximiliano Levi on 03/03/2021.
 */

#ifndef RAYTRACER_TRIANGLE_LIST_H
#define RAYTRACER_TRIANGLE_LIST_H


#include "volume.h"
#include "triangle.h"

class TriangleList : public Volume {
public:
    TriangleList(std::unique_ptr<Triangle[]> triangles, uint32_t count);

    bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const override;
    bool BoundingBox(AABB& bounding_box) const override;
    void Translate(Vector3 offset);
    void Scale(Vector3 scale);

    [[nodiscard]] inline uint32_t Size() const { return count_; }

private:
    std::unique_ptr<Triangle[]> triangles_;
    uint32_t count_;
};


#endif //RAYTRACER_TRIANGLE_LIST_H
