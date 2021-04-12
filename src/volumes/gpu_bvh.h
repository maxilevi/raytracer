/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#include "../kernel/helper.h"
#include "../math/ray.h"
#include "hit_result.h"
#include "aabb.h"
#include "bvh.h"
#include <vector>
#include "gpu_triangle.h"

#ifndef RAYTRACER_GPU_BVH_H
#define RAYTRACER_GPU_BVH_H

class GPUBvhNode {
public:
    GPUBvhNode() = default;
    GPUBvhNode(Bvh* bvh, int left, int right, bool nodes_are_triangles)
    {
        box_ = bvh->box_;
        left_child = left;
        right_child = right;
        has_triangle_nodes = nodes_are_triangles;
    }

    AABB box_;
    int left_child;
    int right_child;
    bool has_triangle_nodes;

    CUDA_DEVICE bool Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const;
};

class GPUBvh {
public:
    CUDA_DEVICE bool Hit(const Ray& ray, double t_min, double t_max, HitResult& record) const;
    static GPUBvh FromBvh(Bvh* cpu_bvh);
    static void Delete(GPUBvh bvh);

private:
    GPUBvh() : node_count(0), triangle_count(0), bvh_nodes(nullptr), gpu_triangles(nullptr) {};
    static int BvhDfs(std::vector<GPUBvhNode>& nodes, std::vector<GPUTriangle>& tris, Bvh* cpu_bvh);

    size_t starting_node;
    size_t node_count;
    size_t triangle_count;
    GPUBvhNode* bvh_nodes;
    GPUTriangle* gpu_triangles;
};


#endif //RAYTRACER_GPU_BVH_H
