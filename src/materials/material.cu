/*
 * Created by Maximiliano Levi on 4/14/2021.
 */

#include "material.h"
#include "../io/stb_image.h"

Material::Material(const char *filename)
{
    int w, h, n;
    unsigned char *data = stbi_load(filename, &w, &h, &n, 3);
    auto colors = new Vector3[w * h * 3];
    int k = 0;
    for(size_t i = 0; i < w; ++i)
    {
        for(size_t j = 0; j < h; ++j)
        {
            colors[i * h + j] = Vector3(data[k++], data[k++], data[k++]);
        }
    }
    this->texture_ = std::unique_ptr<Vector3[]>(colors);
}
