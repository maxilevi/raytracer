/*
 * Created by Maximiliano Levi on 4/15/2021.
 */

#ifndef RAYTRACER_SCENE_LOADER_H
#define RAYTRACER_SCENE_LOADER_H

#include "scene.h"
#include "../io/fast_obj.h"
#include "../materials/material.h"
#include <assert.h>

class SceneLoader {
public:

    static std::pair<std::unique_ptr<Triangle[]>, size_t> LoadOBJ(fastObjMesh* mesh)
    {
        std::vector<std::shared_ptr<Material>> materials;
        std::vector<Triangle> triangles;

        for (size_t i = 0; i < mesh->material_count; ++i) {
            materials.push_back(std::make_shared<Material>(mesh->materials[i].map_Kd.path));
            std::cout << "Loading material \"" << mesh->materials[i].map_Kd.path << "\"" << std::endl;
        }
        for (size_t i = 0; i < mesh->group_count; ++i)
        {
            auto group = mesh->groups[i];
            size_t idx = 0;
            for(size_t j = 0; j < group.face_count; ++j)
            {
                auto fv = mesh->face_vertices[group.face_offset + j];
                auto mat_index = mesh->face_materials[group.index_offset + j];
                assert(fv == 3);
                Vector3 vertices[3], normals[3], uvs[2];
                for(size_t k = 0; k < fv; ++k)
                {
                    auto index = mesh->indices[group.index_offset + idx];

                    vertices[k] = Vector3(mesh->positions[3 * index.p + 0],
                                          mesh->positions[3 * index.p + 1],
                                          mesh->positions[3 * index.p + 2]);
                    normals[k] = Vector3(mesh->normals[3 * index.n + 0],
                                         mesh->normals[3 * index.n + 1],
                                         mesh->normals[3 * index.n + 2]);
                    uvs[0][k] = mesh->texcoords[2 * index.t + 0];
                    uvs[1][k] = 1.0 - mesh->texcoords[2 * index.t + 1];

                    idx++;
                }

                triangles.emplace_back(
                        vertices[0],
                        vertices[1],
                        vertices[2],
                        normals[0],
                        normals[1],
                        normals[2],
                        uvs[0],
                        uvs[1],
                        materials[mat_index]
                );
            }
        }
        auto ptr = new Triangle[triangles.size()];

        for(size_t i = 0; i < triangles.size(); ++i)
            ptr[i] = Triangle(triangles[i]);

        return std::make_pair(std::unique_ptr<Triangle[]>(ptr), triangles.size());

    }

    static std::shared_ptr<TriangleModel> LoadModel(const char* path)
    {
        std::vector<Triangle> triangles;
        fastObjMesh* mesh = fast_obj_read(path);

        auto pair = LoadOBJ(mesh);
        std::cout << "Loaded " << pair.second << " triangles" << std::endl;
        auto model = std::shared_ptr<TriangleModel>(new TriangleModel(std::move(pair.first), pair.second));

        fast_obj_destroy(mesh);
        return model;
    }

    static Scene LouisXIVScene(bool high_quality)
    {
        Scene scene;

        auto model = LoadModel("./../models/louis/low_louis.obj");//high_quality ? "./../models/louis/high_louis.obj" : "./../models/louis/low_louis.obj");

        model->Scale(Vector3(1));
        model->Transform(Matrix3::FromEuler({-3, 0, 0}));
        model->Translate(Vector3(-0.5, -5.5, -4.5));

        scene.Add(model);
        scene.BuildBvh();

        return scene;
    }

    static Scene MarcusAureliusScene(bool high_quality)
    {
        return Scene();
    }

    static Scene IcosphereScene()
    {
        return Scene();
    }

    static Scene TorusScene()
    {
        return Scene();
    }
};

#endif //RAYTRACER_SCENE_LOADER_H
