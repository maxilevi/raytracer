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

        for (size_t i = 0; i < mesh->material_count; ++i)
            materials.push_back(std::make_shared<Material>(mesh->materials[i].map_Kd.path));

        for (size_t i = 0; i < mesh->group_count; ++i)
        {
            auto group = mesh->groups[i];
            for(size_t j = 0; j < group.face_count; ++j)
            {
                auto index = mesh->indices[group.index_offset + j];
                auto mat_index = mesh->face_materials[group.index_offset + j];
                assert(mesh->face_vertices[index.p] == 3);

                Vector3 vertices[3], normals[3], uvs[2];
                for(int w = 0; w < 3; ++w)
                {
                    vertices[w] = Vector3(mesh->positions[index.p + w * 3 + 0],
                                          mesh->positions[index.p + w * 3 + 1],
                                          mesh->positions[index.p + w * 3 + 2]);
                    normals[w] = Vector3(mesh->normals[index.n + w * 3 + 0],
                                         mesh->normals[index.n + w * 3 + 1],
                                         mesh->normals[index.n + w * 3 + 2]);
                    uvs[0][w] = mesh->texcoords[index.t + w * 2 + 0];
                    uvs[1][w] = mesh->texcoords[index.t + w * 2 + 1];
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

    static Scene LoadModel(const char* path)
    {
        Scene scene;
        std::vector<Triangle> triangles;
        fastObjMesh* mesh = fast_obj_read(path);

        auto pair = LoadOBJ(mesh);
        scene.Add(std::make_shared<TriangleModel>(std::move(pair.first), pair.second));

        fast_obj_destroy(mesh);
        return scene;
    }

    static Scene LouisXIVScene(bool high_quality)
    {
        Scene scene = LoadModel(high_quality ? "./../models/louis/high_louis.obj" : "./../models/louis/low_louis.obj");
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
