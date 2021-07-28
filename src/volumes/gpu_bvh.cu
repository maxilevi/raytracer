/*
 * Created by Maximiliano Levi on 4/11/2021.
 */

#include "gpu_bvh.h"

/*
 * When doing a DFS we have in memory at most "H" nodes in the stack, with H = height(tree). We know that the BVH is a binary tree,
 * therefore H = log2(N) with N being the nodes in the tree. Setting a fixed stack with a value of 64 will allow us to process
 * trees of up to 2^64 = 1.84467441e19 nodes.
 */
const int MAX_STACK_SIZE = 64;

CUDA_DEVICE bool GPUBvh::Hit(const Ray &ray, double t_min, double t_max, HitResult &record) const
{

    /*
     * We use a fixed array to simulate a stack and we use the stack to do an iterative DFS. While we have nodes
     * we pop one from the stack and test a ray against it. If we intersect then we have to explore the children so
     * we append those to the stack.
     * */
    GPUBvhNode stack[MAX_STACK_SIZE];
    size_t count = 0;

    /* Add root node */
    stack[count++] = this->bvh_nodes[this->root_index];
    double closest_so_far = t_max;
    bool any_hit = false;

    while (count > 0) {
        /* Pop the last node from the stack */
        GPUBvhNode node = stack[--count];

        if (!node.Hit(ray, t_min, closest_so_far))
            continue;

        if (!node.is_leaf) {
            /* Add both nodes to the stack for further processing */
            GPUBvhNode left = this->bvh_nodes[node.left_child];
            GPUBvhNode right = this->bvh_nodes[node.right_child];

            stack[count++] = right;
            stack[count++] = left;
        } else {
            /* It's a child node, we should test against the triangles and save the closest */
            GPUTriangle left = this->gpu_triangles[node.left_child];
            GPUTriangle right = this->gpu_triangles[node.right_child];

            bool left_hit = left.Hit(ray, t_min, closest_so_far, record);
            closest_so_far = left_hit ? record.t : closest_so_far;
            bool right_hit = right.Hit(ray, t_min, closest_so_far, record);
            if (left_hit || right_hit) {
                any_hit = true;
                closest_so_far = right_hit ? record.t : closest_so_far;
            }
        }
    }
    return any_hit;
}

/* An intersection between a node and a ray is the intersection with it's bounding box (it contains all it's child shapes) */
CUDA_DEVICE bool GPUBvhNode::Hit(const Ray &ray, double t_min, double t_max) const
{
    return this->box_.Hit(ray, t_min, t_max);
}

int GPUBvh::AddTriangle(Triangle *tri, std::vector<GPUTriangle> &tris, std::unordered_map<int, int> &mats_index_map,
                        std::vector<GPUMaterial> &mats)
{
    auto material_id = tri->GetMaterial()->Id();
    auto material_index = -1;
    auto it = mats_index_map.find(material_id);

    /* GPUMaterials are allocated on the GPU therefore we should only create 1 instance for each Material we see */
    if (it == mats_index_map.end()) {
        auto gpu_mat = tri->GetMaterial()->MakeGPUMaterial();
        mats.push_back(gpu_mat);
        material_index = mats.size() - 1;
        mats_index_map.insert({material_id, material_index});
    } else {
        material_index = it->second;
    }

    /* Append the triangle to the list */
    tris.emplace_back(tri, mats[material_index]);
    return (tris.size() - 1);
}

/* Traverse the CPU BVH using DFS and fill the structures as we see the nodes */
int GPUBvh::TraverseBvh(Bvh *node, std::vector<GPUBvhNode> &nodes, std::vector<GPUTriangle> &tris,
                        std::vector<GPUMaterial> &mats, std::unordered_map<int, int> &mats_index_map)
{
    auto *left_child = node->left_.get();
    auto *right_child = node->right_.get();
    /* In our BVH the objects are sorted and the left and right child represent the extremes of the collection. Therefore
     * with simple pointer arithmetic we can retrieve the amount of objects the BVH has.
     */
    size_t objects_left = node->end_ - node->start_;

    /* We have 3 cases. Either the node contains 1 or 2 triangles or it contains other nodes */
    if (objects_left == 1) {
        int index = AddTriangle((Triangle *) left_child, tris, mats_index_map, mats);

        nodes.emplace_back(node, index, index, true);
    } else if (objects_left == 2) {
        int index1 = AddTriangle((Triangle *) left_child, tris, mats_index_map, mats);
        int index2 = AddTriangle((Triangle *) right_child, tris, mats_index_map, mats);

        nodes.emplace_back(node, index1, index2, true);
    } else {
        /* Recursively add both children and use the returned indexes for the parent node */
        auto left_index = TraverseBvh((Bvh *) left_child, nodes, tris, mats, mats_index_map);
        auto right_index = TraverseBvh((Bvh *) right_child, nodes, tris, mats, mats_index_map);

        nodes.emplace_back(node, left_index, right_index, false);
    }
    /* Return the index of the node we just processed */
    return (int) (nodes.size() - 1);
}

/* Traverse the BVH and retrieve it's data and then allocate a GPUBVH on GPU memory */
std::pair<GPUBvh, std::vector<GPUMaterial>> GPUBvh::FromBvh(Bvh *cpu_bvh)
{
    /* Retrieve the data */
    std::vector<GPUBvhNode> nodes;
    std::vector<GPUTriangle> tris;
    std::vector<GPUMaterial> mats;
    std::unordered_map<int, int> mats_index_map;
    auto root_index = TraverseBvh(cpu_bvh, nodes, tris, mats, mats_index_map);

    /* Now create the BVH */
    GPUBvh bvh;
    bvh.root_index = root_index;
    bvh.triangle_count = tris.size();
    bvh.node_count = nodes.size();

    std::cout << "Allocating GPU Bvh " << std::endl;

    /* Finally allocate the memory on the CPU and copy to it. */
    CUDA_CALL(cudaMalloc(&bvh.bvh_nodes, sizeof(GPUBvhNode) * nodes.size()));
    CUDA_CALL(cudaMalloc(&bvh.gpu_triangles, sizeof(GPUTriangle) * tris.size()));

    CUDA_CALL(cudaMemcpy(bvh.bvh_nodes, &nodes[0], sizeof(GPUBvhNode) * nodes.size(), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(bvh.gpu_triangles, &tris[0], sizeof(GPUTriangle) * tris.size(), cudaMemcpyHostToDevice));

    return std::make_pair(bvh, mats);
}

/* Delete the GPU memory allocated for this BVH */
void GPUBvh::Delete(GPUBvh bvh)
{
    CUDA_CALL(cudaFree(bvh.bvh_nodes));
    CUDA_CALL(cudaFree(bvh.gpu_triangles));
}
