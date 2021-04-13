/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#include "gpu_bvh.h"
#include "../kernel/vector.h"


CUDA_DEVICE bool GPUBvh::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{
    /*
    HitResult temp;
    bool any_hit = false;
    double closest_so_far = t_max;
    for (size_t i = 0; i < this->triangle_count; ++i)
    {
        if(this->gpu_triangles[i].Hit(ray, t_min, closest_so_far, temp))
        {
            any_hit = true;
            closest_so_far = temp.t;
            record = temp;
        }
    }
    return any_hit;*/

    /*
     * I chose to use a fixed size stack in order to have a "queue" like behaviour and to avoid unnecessary allocations which slow down the kernel.
     * A queue would have made more sense but a stack was easier to implement
     * */
    const int MAX_STACK_SIZE = 24;
    GPUBvhNode stack[MAX_STACK_SIZE];
    size_t count = 0;

    /* Add starting element */
    stack[count++] = this->bvh_nodes[this->starting_node];
    double closest_so_far = t_max;
    bool any_hit = false;
    while(count > 0)
    {
        GPUBvhNode node = stack[--count];

        if (!node.Hit(ray, t_min, closest_so_far))
            continue;

        if (node.has_triangle_nodes)
        {
            GPUTriangle left = this->gpu_triangles[node.left_child];
            GPUTriangle right = this->gpu_triangles[node.right_child];

            bool left_hit = left.Hit(ray, t_min, closest_so_far, record);
            closest_so_far = left_hit ? record.t : closest_so_far;
            bool right_hit = right.Hit(ray, t_min, closest_so_far, record);
            if (left_hit || right_hit)
            {
                any_hit = true;
                closest_so_far = right_hit ? record.t : closest_so_far;
            }
        }
        else {
            GPUBvhNode left = this->bvh_nodes[node.left_child];
            GPUBvhNode right = this->bvh_nodes[node.right_child];

            stack[count++] = right;
            stack[count++] = left;
        }
    }
    return any_hit;
}

CUDA_DEVICE bool GPUBvhNode::Hit(const Ray &ray, double t_min, double t_max) const
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

        nodes.emplace_back(nodes.size(), cpu_bvh, (int)(tris.size()-1), (int)(tris.size()-1), true);
    }
    else if (objects_left == 2)
    {
        auto* tri1 = (Triangle*)left_child;
        auto* tri2 = (Triangle*)right_child;

        tris.emplace_back(tri1);
        tris.emplace_back(tri2);

        nodes.emplace_back(nodes.size(), cpu_bvh, (int)(tris.size()-2), (int)(tris.size()-1), true);
    }
    else {
        auto left_idx = BvhDfs(nodes, tris, (Bvh *) left_child);
        auto right_idx = BvhDfs(nodes, tris, (Bvh *) right_child);

        nodes.emplace_back(nodes.size(), cpu_bvh, left_idx, right_idx, false);
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
