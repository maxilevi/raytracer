/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#ifndef RAYTRACER_MATERIAL_H
#define RAYTRACER_MATERIAL_H
#include "../math/ray.h"
#include "../volumes/hit_result.h"

typedef unsigned char uchar_t;

class Material {
public:
    Material() {};
    Material(const char* texture);
    CUDA_HOST_DEVICE ~Material();
    CUDA_HOST_DEVICE Vector3 Sample(double s, double t) const;
    CUDA_HOST_DEVICE Vector3 BilinearSample(double s, double t) const;
    int Id();
    Material MakeGPUMaterial();
    void FreeGPUMaterial();
    bool Scatter(const Ray&, const HitResult&, Vector3& attenuation, Ray&) const;

private:
    static int ID_COUNTER;
    int id_;
    bool is_in_gpu_;
    uchar_t* texture_;
    size_t width_;
    size_t height_;
    double texel_width_;
    double texel_height_;
};

typedef Material GPUMaterial;

#endif //RAYTRACER_MATERIAL_H
