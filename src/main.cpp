#include <iostream>
#include <fstream>
#include "Vector3.h"
#include "Camera.h"
#include "tga.h"

void WriteOutput(const std::string& path, const Camera& camera)
{
    constexpr uint8_t channels = 3;
    auto bgr_frame = std::unique_ptr<uint8_t[]>(new uint8_t[camera.Width() * camera.Height() * channels]);
    auto* frame = camera.GetFrame();
    for(uint32_t i = 0; i < camera.Width(); ++i)
    {
        for(uint32_t j = 0; j < camera.Height(); ++j)
        {
            for(uint32_t c = 0; c < channels; ++c)
            {
                bgr_frame[j * camera.Width() * channels + i * channels + c] = (uint8_t) (frame[j * camera.Width() + i][c] * 255);
            }
        }
    }
    TGAWrite(path, camera.Width(), camera.Height(), bgr_frame.get(), channels);
}

int main()
{
    Camera camera(200, 100);

    camera.Draw();

    WriteOutput("./output.tga", camera);

    return 0;
}