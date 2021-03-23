/*
 * Created by Maximiliano Levi on 02/03/2021.
 */

#ifndef RAYTRACER_AABB_H
#define RAYTRACER_AABB_H

#include "../math/vector3.h"
#include "../math/ray.h"

class AABB {
public:
    AABB() = default;
    CUDA_DEVICE AABB(const Vector3& min, const Vector3& max) : min_(min), max_(max) {};

    [[nodiscard]] inline const Vector3& Min() const { return min_; }
    [[nodiscard]] inline const Vector3& Max() const { return max_; }

    CUDA_DEVICE inline bool Hit(const Ray& ray, double t_min, double t_max) const
    {
        for(int i = 0; i < 3; ++i)
        {
            auto inverse_direction = 1.0 / ray.Direction()[i];
            auto t0 = (min_[i] - ray.Origin()[i]) * inverse_direction;
            auto t1 = (max_[i] - ray.Origin()[i]) * inverse_direction;
            if (inverse_direction < 0.0) {
                auto tmp = t1;
                t1 = t0;
                t0 = tmp;
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    CUDA_DEVICE static AABB Merge(const AABB& a, const AABB& b);

private:
    Vector3 min_;
    Vector3 max_;

};


#endif //RAYTRACER_AABB_H
