/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#include "aabb.h"

AABB AABB::Merge(const AABB &a, const AABB &b)
{
    Vector3 min, max;

    for(int i = 0; i < 3; ++i)
    {
        min[i] = MIN(a.min_[i], b.min_[i]);
        max[i] = MAX(a.max_[i], b.max_[i]);
    }

    return {min, max};
}