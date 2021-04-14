#include "kernel/helper.h"
#include <iostream>
#include "math/vector3.h"
#include "camera.h"
#include "io/tga.h"
#include "io/ply.h"
#include <chrono>
#include <string>
#include <cstdint>
#include "helper.h"
#include "backends/gpu_backend.h"
#include "backends/cpu_backend.h"

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

std::unique_ptr<RenderingBackend> GetBackend(int argc, char** argv)
{
    if (argc > 1)
    {
        for(size_t i = 0; i < argc; ++i)
        {
            auto str = std::string(argv[i]);
            auto name = str.substr(0, 9);
            if (name == "--backend")
            {
                auto val = str.substr(10, 3);
                if (val == "GPU" || val == "CPU") {
                    std::cout << "Selected backend \"" << val << "\" " << std::endl;
                    return (val == "GPU") ? (std::unique_ptr<RenderingBackend>) std::make_unique<GPUBackend>() : (std::unique_ptr<RenderingBackend>) std::make_unique<CPUBackend>();
                }
                else {
                    break;
                }
            }
        }
    }
    std::cout << "No backend supplied, using default \"GPU\"." << std::endl;
    return std::make_unique<GPUBackend>();
}

int LoadScene(Scene& scene, std::chrono::time_point<std::chrono::steady_clock> t1)
{
        std::shared_ptr<TriangleModel> model = LoadPLY("./../models/torus.ply");

    std::cout << "Loaded " << model->Size() << " triangles" << std::endl;
    std::cout << "Loading the model took " << TimeIt(t1) << " ms" << std::endl;
    if(model == nullptr) return 1;

    // Icosphere
    model->Scale(Vector3(0.25));
    model->Transform(Matrix3::FromEuler({0, 90, 0}));
    model->Translate(Vector3(0, 0, -0.5));

    // Torus
    //model->Scale(Vector3(1));
    //model->Transform(Matrix3::FromEuler({0, 90, 0}));
    //model->Translate(Vector3(0, 0, -0.5));

    // Statue
    //model->Scale(Vector3(1));
    //model->Transform(Matrix3::FromEuler({0, 0, 0}));
    //model->Translate(Vector3(0, 0, -0.5));

    scene.Add(model);
    std::cout << "Making scene took " << TimeIt(t1) << " ms" << std::endl;

    scene.BuildBvh();
    std::cout << "Building the bvh took " << TimeIt(t1) << " ms" << std::endl;

    return 0;
}

int main(int argc, char** argv)
{
    Scene scene;

    auto t1 = std::chrono::high_resolution_clock::now();

    int r;
    if ((r = LoadScene(scene, t1)))
        return r;

    /* Backend */
    std::unique_ptr<RenderingBackend> backend = GetBackend(argc, argv);
    if (!backend)
        return 1;

    /* Camera */
    Camera camera(1920, 1080, backend);

    camera.Draw(scene);

    std::cout << "Drawing took " << TimeIt(t1) / 1000 << " s" << std::endl;

    WriteOutput("./output.tga", camera);

    return 0;
}