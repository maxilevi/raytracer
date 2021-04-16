/*
 * Created by Maximiliano Levi on 03/03/2021.
 */

#ifndef RAYTRACER_TRIANGLE_MODEL_H
#define RAYTRACER_TRIANGLE_MODEL_H


#include "../volumes/triangle.h"
#include "../math/matrix3.h"

class TriangleModel  {
public:
    TriangleModel(std::unique_ptr<Triangle[]> triangles, uint32_t count);

    void Translate(Vector3 offset);
    void Scale(Vector3 scale);
    void Transform(Matrix3 transformation);

    static std::unique_ptr<TriangleModel> Merge(const TriangleModel*, const TriangleModel*);

    [[nodiscard]] inline uint32_t Size() const { return count_; }

    std::unique_ptr<Triangle[]> triangles_;
private:
    uint32_t count_;
};


#endif //RAYTRACER_TRIANGLE_MODEL_H
