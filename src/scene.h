/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#ifndef RAYTRACER_SCENE_H
#define RAYTRACER_SCENE_H


#include "volume.h"
#include <vector>

class Scene {
public:
    bool Add(Volume*);
    bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const;
    Volume* operator[](int idx);
    uint64_t Size() const { return volumes_.size(); }

private:
    std::vector<Volume*> volumes_;
};


#endif //RAYTRACER_SCENE_H
