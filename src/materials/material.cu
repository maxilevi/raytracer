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
    unsigned char *data = stbi_load(filename, &w, &h, &n, 3);
    assert(w != 0);
    assert(h != 0);
    assert(n == 3);
    auto colors = new uint8_t[w * h * 3];
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
    if (s < 0.0 || s > 1.0)
        printf("error %f", s);
    if (t < 0.0 || t > 1.0)
        printf("error %f", t);
    auto x = size_t(s * width_);
    auto y = size_t(t * height_);

    Vector3 ans;
    for(auto i = 0; i < 3; ++i)
    {
        //Modulo(ref X) * _boundsY * _boundsZ + Y * _boundsZ + Modulo(ref Z)
        auto idx = y * width_ * 3 + x * 3 + i;//y * width_ * 3 + x * 3 + i;
        ans[i] = this->texture_[idx];
    }
    return ans / 256.0;
    //printf("%d, %d %d\n", (int)offset, (int)width_, (int)height_);
    //return Vector3(this->texture_[offset + 0], this->texture_[offset + 1], this->texture_[offset + 2]) / 256.0;
}

CUDA_HOST_DEVICE Vector3 Material::BilinearSample(double s, double t) const
{
    auto floored_s = (size_t(s * width_) / double(width_));
    auto floored_t = (size_t(t * height_) / double(height_));
    auto x = (s - floored_s) / texel_width_;
    auto y = (t - floored_t) / texel_height_;

    auto center = this->Sample(s, t);
    auto top_left = this->Sample(s - texel_width_, t + texel_height_);
    auto bot_left = this->Sample(s - texel_width_, t - texel_height_);
    auto top_right = this->Sample(s + texel_width_, t + texel_height_);
    auto bot_right = this->Sample(s + texel_width_, t - texel_height_);

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

    CUDA_CALL(cudaMalloc(&mat.texture_, sizeof(uint8_t) * width_ * height_ * 3));
    CUDA_CALL(cudaMemcpy(mat.texture_, texture_, sizeof(uint8_t) * width_ * height_ * 3, cudaMemcpyHostToDevice));
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
