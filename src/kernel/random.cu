/*
 * Created by Maximiliano Levi on 4/9/2021.
 */

#include "random.h"

__host__ __device__ double RandomDouble(uint32_t& seed)
{
    int k;
    int s = int(seed);
    if (s == 0)
        s = 305420679;
    k = s / 127773;
    s = 16807 * (s - k * 127773) - 2836 * k;
    if (s < 0)
        s += 2147483647;
    seed = uint32_t(s);
    return double(seed % uint32_t(65536)) / 65535.0;
}

__host__ __device__ int RandomInt(uint32_t& seed, int min, int max)
{
    return RandomDouble(seed) * max + min;
}