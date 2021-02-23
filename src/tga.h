/*
 * Created by Maximiliano Levi on 22/02/2021.
 */

#ifndef RAYTRACER_TGA_H
#define RAYTRACER_TGA_H


#include <cstdint>
#include <fstream>

const uint32_t kTGAHeaderSize = 18;

typedef struct {
    char idlength;
    char colourmaptype;
    char datatypecode;
    short int colourmaporigin;
    short int colourmaplength;
    char colourmapdepth;
    short int x_origin;
    short int y_origin;
    short width;
    short height;
    char bitsperpixel;
    char imagedescriptor;
} TGAHeader;

void TGAWrite(const std::string& filename, uint32_t width, uint32_t height, uint8_t* data_bgr, uint8_t channels)
{
    std::ofstream file;
    file.open(filename, std::ofstream::binary | std::ofstream::out);

    uint8_t header[kTGAHeaderSize] = {0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    header[12] = (uint8_t)(width & 0xFFu);
    header[13] = (uint8_t) ((width >> 8u) & 0xFFu);
    header[14] = (uint8_t) (height & 0xFFu);
    header[15] = (uint8_t) ((height >> 8u) & 0xFFu);
    header[16] = 24;

    for (uint32_t i = 0; i < kTGAHeaderSize; ++i)
        file << header[i];

    for(uint32_t i = 0; i < width; ++i)
    {
        for(uint32_t j = 0; j < height; ++j)
        {
            uint32_t idx = i * height * channels + j * channels;
            for(int n = 0; n < channels; n++)
                file << data_bgr[idx + n];
        }
    }
    file.close();
}

#endif //RAYTRACER_TGA_H
