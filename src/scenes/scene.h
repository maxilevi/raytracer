/*
 * Created by Maximiliano Levi on 23/02/2021.
 */

#ifndef RAYTRACER_SCENE_H
#define RAYTRACER_SCENE_H


#include "../volumes/triangle.h"
#include "../io/triangle_model.h"
#include "../volumes/hit_result.h"
#include "../volumes/aabb.h"
#include "../volumes/bvh.h"

class Scene {
public:
    Scene() : model_(nullptr) {};
    void Add(std::shared_ptr<TriangleModel>);
    void BuildBvh();
    inline Bvh* GetBvh() const
    {
        return bvh_.get();
    }

private:
    std::unique_ptr<TriangleModel> model_;
    std::unique_ptr<Bvh> bvh_;
};


#endif //RAYTRACER_SCENE_H
