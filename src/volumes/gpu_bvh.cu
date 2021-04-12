/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#include "gpu_bvh.h"
#include "../kernel/vector.h"

CUDA_DEVICE bool GPUBvh::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    /*
    for(int i = 0; i < this->triangle_count; ++i)
        if (this->gpu_triangles[i].Hit(ray, t_min, t_max, record))
            return true;
    return false;
     */
    vector<GPUBvhNode> stack;
    stack.push_back(this->bvh_nodes[this->starting_node]);

    while(!stack.empty())
    {
        GPUBvhNode node = stack.pop();
        if (!node.Hit(ray, t_min, t_max, record))
            continue;

        if (node.has_triangle_nodes)
        {
            GPUTriangle left = this->gpu_triangles[node.left_child];
            GPUTriangle right = this->gpu_triangles[node.right_child];

            return left.Hit(ray, t_min, t_max, record) || right.Hit(ray, t_min, t_max, record);
        }
        else {
            GPUBvhNode left = this->bvh_nodes[node.left_child];
            GPUBvhNode right = this->bvh_nodes[node.right_child];

            stack.push_back(left);
            stack.push_back(right);
        }
    }
    return false;
}

CUDA_DEVICE bool GPUBvhNode::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    return this->box_.Hit(ray, t_min, t_max);
}

int GPUBvh::BvhDfs(std::vector<GPUBvhNode>& nodes, std::vector<GPUTriangle>& tris, Bvh* cpu_bvh)
{
    size_t objects_left = cpu_bvh->end_ - cpu_bvh->start_;
    auto* left_child = cpu_bvh->left_.get();
    auto* right_child = cpu_bvh->right_.get();
    if (objects_left == 1)
    {
        auto* tri = (Triangle*)left_child;
        tris.emplace_back(tri);

        nodes.emplace_back(cpu_bvh, tris.size()-1, tris.size()-1, true);
    }
    else if (objects_left == 2)
    {
        auto* tri1 = (Triangle*)left_child;
        auto* tri2 = (Triangle*)right_child;

        tris.emplace_back(tri1);
        tris.emplace_back(tri2);

        nodes.emplace_back(cpu_bvh, tris.size()-2, tris.size()-1, true);
    }
    else {
        auto left_idx = BvhDfs(nodes, tris, (Bvh *) left_child);
        auto right_idx = BvhDfs(nodes, tris, (Bvh *) right_child);

        nodes.emplace_back(cpu_bvh, left_idx, right_idx, false);
    }

    return nodes.size()-1;
}

GPUBvh GPUBvh::FromBvh(Bvh* cpu_bvh)
{
    std::vector<GPUBvhNode> nodes;
    std::vector<GPUTriangle> tris;
    auto starting_node = BvhDfs(nodes, tris, cpu_bvh);

    GPUBvh bvh;
    bvh.starting_node = starting_node;
    bvh.triangle_count = tris.size();
    bvh.node_count = nodes.size();

    std::cout << "Allocating Bvh" << std::endl;

    CUDA_CALL(cudaMalloc(&bvh.bvh_nodes, sizeof(GPUBvhNode) * nodes.size()));
    CUDA_CALL(cudaMalloc(&bvh.gpu_triangles, sizeof(GPUTriangle) * tris.size()));

    CUDA_CALL(cudaMemcpy(bvh.bvh_nodes, &nodes[0], sizeof(GPUBvhNode) * nodes.size(), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(bvh.gpu_triangles, &tris[0], sizeof(GPUTriangle) * tris.size(), cudaMemcpyHostToDevice));

    return bvh;
}

void GPUBvh::Delete(GPUBvh bvh)
{
    CUDA_CALL(cudaFree(bvh.bvh_nodes));
    CUDA_CALL(cudaFree(bvh.gpu_triangles));
}
