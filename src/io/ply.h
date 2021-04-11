/*
 * Created by Maximiliano Levi on 28/02/2021.
 */

#ifndef RAYTRACER_PLY_H
#define RAYTRACER_PLY_H

#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <regex>
#include "../volumes/triangle.h"
#include "triangle_model.h"

bool ProcessHeader(const std::string& line, bool& is_header, int& vertex_count, int& triangle_count, bool& has_vertices, bool& has_normals, bool& has_uvs)
{
    auto iss = std::istringstream(line);
    std::string token, token2, token3;
    iss >> token;
    if (token == "ply" || token == "comment") {
        /* Skip */
    }
    else if (token == "format") {
        iss >> token2;
        if (token2 != "ascii") {
            std::cout << "Non ascii PLY files are not supported." << std::endl;
            return false;
        }
    }
    else if (token == "element") {
        iss >> token2;
        if (token2 == "vertex") {
            iss >> token3;
            vertex_count = std::atoi(token3.c_str());
        } else if (token2 == "face") {
            iss >> token3;
            triangle_count = std::atoi(token3.c_str());
        }
    }
    else if (token == "property") {
        iss >> token2;
        if (token2 == "float") {
            iss >> token3;
            has_vertices |= (token3 == "x" || token3 == "y" || token3 == "z");
            has_normals |= (token3 == "x" || token3 == "y" || token3 == "z");
        }
    }
    else if (token == "end_header") {
        is_header = false;
    }
    return true;
}

bool ProcessBody(const std::string& line, std::vector<Vector3>& vertices, std::vector<Vector3>& normals, std::unique_ptr<Triangle[]>& triangles, int& triangle_index,
                 std::string* token_buffer, const int& tokens_per_line, const bool& has_vertices, const bool& has_normals, const bool& has_uvs, bool& is_reading_verts, const int& vertex_count)
{
    auto iss = std::istringstream(line);
    if (is_reading_verts)
    {
        if (!has_vertices) {
            std::cout << "Model has no vertices" << std::endl;
            return false;
        }

        for(int i = 0; i < tokens_per_line; ++i)
            iss >> token_buffer[i];

        vertices.emplace_back(std::atof(token_buffer[0].c_str()), std::atof(token_buffer[1].c_str()), std::atof(token_buffer[2].c_str()));

        if (has_normals)
            normals.emplace_back(std::atof(token_buffer[3].c_str()), std::atof(token_buffer[4].c_str()), std::atof(token_buffer[5].c_str()));

        if (vertices.size() == vertex_count)
            is_reading_verts = false;
    }
    else {
        std::string count_str, s0, s1, s2;
        int count, i0, i1, i2;

        iss >> count_str;
        count = std::atoi(count_str.c_str());
        if (count != 3)
        {
            std::cout << "Only triangles are supported." << std::endl;
            return false;
        }
        iss >> s0;
        iss >> s1;
        iss >> s2;
        i0 = std::atoi(s0.c_str());
        i1 = std::atoi(s1.c_str());
        i2 = std::atoi(s2.c_str());

        triangles[triangle_index++] = (has_normals)
                ? Triangle(vertices[i0], vertices[i1], vertices[i2])
                : Triangle(vertices[i0], vertices[i1], vertices[i2], normals[i0], normals[i1], normals[i2]);
    }
    return true;
}


std::unique_ptr<TriangleModel> LoadPLY(const std::string& path)
{
    std::unique_ptr<Triangle[]> triangles;
    std::string token_buffer[9];
    std::vector<Vector3> vertices, normals;
    std::ifstream infile(path);
    std::string line;
    bool is_header = true, has_normals = false, has_vertices = false, has_uvs = false, is_reading_verts = true;
    int vertex_count = 0, triangle_count = 0, triangle_index = 0;

    if (!infile.good()) {
        std::cout << "Failed to load file " << path << std::endl;
        return nullptr;
    }

    while (std::getline(infile, line))
    {
        if(is_header)
        {
            if (!ProcessHeader(line, is_header, vertex_count, triangle_count, has_vertices, has_normals, has_uvs))
                return nullptr;

            if (!is_header)
                triangles = std::unique_ptr<Triangle[]>(new Triangle[triangle_count]);

        } else {
            const int tokens_per_line = (has_vertices && has_normals && has_uvs ? 8 : has_vertices && has_normals ? 6 : 3);
            if (!ProcessBody(line, vertices, normals, triangles, triangle_index, token_buffer, tokens_per_line, has_vertices, has_normals, has_uvs, is_reading_verts, vertex_count))
                return nullptr;
        }
    }
    return std::make_unique<TriangleModel>(std::move(triangles), triangle_count);
}


#endif //RAYTRACER_PLY_H
