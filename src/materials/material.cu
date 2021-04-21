/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#include "material.h"
#include "../io/tga.h"
#include "../io/stb_image.h"
#include <string>
#include <assert.h>

int Material::ID_COUNTER = 0;

Material::Material(const char *filename)
{
    int w, h, n;
    uchar_t *data = stbi_load(filename, &w, &h, &n, 3);
    assert(w != 0);
    assert(h != 0);
    assert(n == 3);
    auto colors = new uchar_t[w * h * 3];
    for(size_t j = 0; j < h * w * 3; ++j)
        colors[j] = data[j];

    stbi_image_free(data);
    this->texture_ = colors;
    this->width_ = w;
    this->height_ = h;
    this->texel_width_ = (1.0 / width_);
    this->texel_height_ = (1.0 / height_);
    this->is_in_gpu_ = false;
    this->id_ = ID_COUNTER++;
}

CUDA_HOST_DEVICE Vector3 Material::Sample(double s, double t) const
{
    auto x = size_t(s * width_);
    auto y = size_t(t * height_);

    Vector3 ans;
    for(auto i = 0; i < 3; ++i)
    {
        auto idx = y * width_ * 3 + x * 3 + i;
        ans[i] = this->texture_[idx];
    }
    //if (ans.X() <= DBL_EPSILON && ans.Y() <= DBL_EPSILON && ans.Z() <= DBL_EPSILON)
    //    printf("Pixels %d %d have no color\n", (int)x, (int)y);
    return ans / 256.0;
}

CUDA_HOST_DEVICE Vector3 Material::BilinearSample(double s, double t) const
{
    auto floored_s = (size_t(s * width_) / double(width_));
    auto floored_t = (size_t(t * height_) / double(height_));
    auto x = (s - floored_s) / texel_width_;
    auto y = (t - floored_t) / texel_height_;
    auto tw = texel_width_;
    auto th = texel_height_;

    auto center = this->Sample(s, t);
    auto top_left = this->Sample(s - tw, t + th);
    auto bot_left = this->Sample(s - tw, t - th);
    auto top_right = this->Sample(s + tw, t + th);
    auto bot_right = this->Sample(s + tw, t - th);

    return Vector3::Lerp(Vector3::Lerp(bot_left, bot_right, x), Vector3::Lerp(top_left, top_right, x), y);
}

CUDA_HOST_DEVICE Material::~Material()
{
    if (!is_in_gpu_)
        delete[] texture_;
}

Material Material::MakeGPUMaterial()
{
    Material mat;
    mat.is_in_gpu_ = true;
    mat.texel_height_ = texel_height_;
    mat.texel_width_ = texel_width_;
    mat.width_ = width_;
    mat.height_ = height_;
    mat.id_ = id_;
    assert(mat.width_ != 0);
    assert(mat.height_ != 0);

    CUDA_CALL(cudaMalloc(&mat.texture_, sizeof(uchar_t) * width_ * height_ * 3));
    CUDA_CALL(cudaMemcpy(mat.texture_, texture_, sizeof(uchar_t) * width_ * height_ * 3, cudaMemcpyHostToDevice));
    return mat;
}

void Material::FreeGPUMaterial()
{
    CUDA_CALL(cudaFree(texture_));
}

int Material::Id()
{
    return id_;
}
/*
bool Material::Scatter(const Ray& , const HitResult & result, Vector3 &attenuation, Ray &) const
{
    auto scatter_direction = rec.normal + random_unit_vector();
    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;

    Vector3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected);
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
}
*/