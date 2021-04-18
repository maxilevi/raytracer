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

void TGAWrite(const std::string& filename, uint32_t width, uint32_t height, uint8_t* data_bgr, uint8_t channels);

#endif //RAYTRACER_TGA_H
