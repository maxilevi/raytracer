/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#ifndef RAYTRACER_SCENE_H
#define RAYTRACER_SCENE_H


#include "volumes/volume.h"
#include <vector>
#include <memory>

class Scene : public Volume {
public:
    bool Add(std::shared_ptr<Volume>);
    bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const override;
    bool BoundingBox(AABB& output_box) const override;

private:
    std::vector<std::shared_ptr<Volume>> volumes_;
};


#endif //RAYTRACER_SCENE_H
