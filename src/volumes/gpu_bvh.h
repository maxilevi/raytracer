/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#include "../kernel/helper.h"
#include "../math/ray.h"
#include "hit_result.h"
#include "aabb.h"
#include "bvh.h"
#include <vector>
#include <unordered_map>
#include "gpu_triangle.h"
#include "../materials/material.h"

#ifndef RAYTRACER_GPU_BVH_H
#define RAYTRACER_GPU_BVH_H

/* A single node for the GPU BVH. A node can either contain triangles or other nodes. */
class GPUBvhNode {
public:
    GPUBvhNode() = default;

    GPUBvhNode(Bvh *bvh, int left, int right, bool is_leaf)
    {
        this->box_ = bvh->box_;
        this->left_child = left;
        this->right_child = right;
        this->is_leaf = is_leaf;
    }

    AABB box_;
    int left_child;
    int right_child;
    bool is_leaf;

    CUDA_DEVICE bool Hit(const Ray &ray, double t_min, double t_max) const;
};

/* BVH for the GPU */
class GPUBvh {
public:
    CUDA_DEVICE bool Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const;

    static std::pair<GPUBvh, std::vector<GPUMaterial>> FromBvh(Bvh *cpu_bvh);

    static void Delete(GPUBvh bvh);

private:
    GPUBvh() : node_count(0), triangle_count(0), bvh_nodes(nullptr), gpu_triangles(nullptr)
    {};

    static int TraverseBvh(Bvh *node, std::vector<GPUBvhNode> &nodes, std::vector<GPUTriangle> &tris,
                           std::vector<GPUMaterial> &mats, std::unordered_map<int, int> &mats_index_map);

    static int AddTriangle(Triangle *tri, std::vector<GPUTriangle> &tris, std::unordered_map<int, int> &mats_index_map,
                           std::vector<GPUMaterial> &mats);

    size_t root_index;
    size_t node_count;
    size_t triangle_count;
    /* These are pointers to GPU memory */
    GPUBvhNode *bvh_nodes;
    GPUTriangle *gpu_triangles;
};


#endif //RAYTRACER_GPU_BVH_H
