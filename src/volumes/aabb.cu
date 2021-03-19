/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#include "aabb.h"
#include <algorithm>

CUDA_CALLABLE_MEMBER AABB AABB::Merge(const AABB &a, const AABB &b)
{
    Vector3 min, max;

    for(int i = 0; i < 3; ++i)
    {
        min[i] = std::min(a.min_[i], b.min_[i]);
        max[i] = std::max(a.max_[i], b.max_[i]);
    }

    return {min, max};
}