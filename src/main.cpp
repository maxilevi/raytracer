#include <iostream>
#include <fstream>
#include "Vector3.h"
#include "Camera.h"
#include "io/tga.h"
#include "scene.h"
#include "volumes/sphere.h"
#include <chrono>

void WriteOutput(const std::string& path, const Camera& camera)
{
    constexpr uint8_t channels = 3;
    auto bgr_frame = std::unique_ptr<uint8_t[]>(new uint8_t[camera.Width() * camera.Height() * channels]);
    auto* frame = camera.GetFrame();
    for(uint32_t i = 0; i < camera.Width(); ++i)
    {
        for(uint32_t j = 0; j < camera.Height(); ++j)
        {

            for(int c_src = channels-1; c_src > -1; --c_src)
            {
                bgr_frame[j * camera.Width() * channels + i * channels + (channels - c_src - 1)] = (uint8_t) (frame[j * camera.Width() + i][c_src] * 255);
            }
        }
    }
    TGAWrite(path, camera.Width(), camera.Height(), bgr_frame.get(), channels);
}

int main()
{
    Scene scene;

    /* Fill scene */
    Sphere sphere(Vector3(0, 0, -1), 0.5);
    scene.Add(&sphere);
    Sphere floor(Vector3(0, -100.5, -1), 100);
    scene.Add(&floor);

    /* Camera */
    Camera camera(480, 270);

    auto t1 = std::chrono::high_resolution_clock::now();
    camera.Draw(scene);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    std::cout << "Drawing took " << duration << " ms";

    WriteOutput("./output.tga", camera);

    return 0;
}