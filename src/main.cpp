#include "defines.h"
#include <iostream>
#include <fstream>
#include "vector3.h"
#include "camera.h"
#include "io/tga.h"
#include "io/ply.h"
#include "volumes/bvh.h"
#include "scene.h"
#include "volumes/sphere.h"
#include "volumes/triangle_list.h"
#include <chrono>
#include <string>
#include <cstdint>

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

auto TimeIt(std::chrono::time_point<std::chrono::steady_clock>& prev_time)
{
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - prev_time ).count();
    prev_time = t2;
    return duration;
}

int LoadScene(Scene& scene, std::chrono::time_point<std::chrono::steady_clock> t1)
{
    Sphere floor(Vector3(0, -100.5, -1), 100);
    scene.Add(std::shared_ptr<Volume>(new Sphere(Vector3(0, -100.5, -1), 100)));

    std::shared_ptr<TriangleList> model = LoadPLY("./../models/icosphere.ply");
    model->Scale(Vector3(0.5));
    model->Translate(Vector3(0, 0, -1));
    scene.Add(model);

    std::cout << "Loaded " << model->Size() << " triangles" << std::endl;
    std::cout << "Loading the model took " << TimeIt(t1) << " ms" << std::endl;

    return 0;
}

int main()
{
    Scene scene;

    auto t1 = std::chrono::high_resolution_clock::now();

    int r;
    if ((r = LoadScene(scene, t1)))
        return r;

    /* Camera */
    Camera camera(1920 / 8, 1080 / 8);

    camera.Draw(scene);

    std::cout << "Drawing took " << TimeIt(t1) << " ms";

    WriteOutput("./output.tga", camera);

    return 0;
}