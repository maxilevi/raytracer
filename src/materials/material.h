/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#ifndef RAYTRACER_MATERIAL_H
#define RAYTRACER_MATERIAL_H
#include "../math/ray.h"
#include "../volumes/hit_result.h"

class UberMaterial {
public:
    bool Scatter(const Ray&, const HitResult&, Vector3& attenuation, Ray&) const;

private:
    
};


#endif //RAYTRACER_MATERIAL_H
