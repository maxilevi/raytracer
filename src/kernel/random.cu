/*
 * Created by Maximiliano Levi on 4/9/2021.
 */

#include "random.h"

/*
__host__ __device__ int Random(uint32_t& seed)
{
    seed = (1664525 * seed + 1013904223) % 2147483648;
    return seed;// & (0xFFFFFFFF);
    /*
    double val = (sin(seed * 12.9898) * 43758.5453);
    seed = (seed + 1) % 113;//((uint32_t) val * 524287) % 113;
    return val - (uint32_t) val;
}

__host__ __device__ double RandomDouble(uint32_t& seed)
{
    //return Random(seed) / 2147483648.0;
    int k;
    int s = int(seed);
    if (s == 0)
        s = 305420679;
    k = s / 127773;
    s = 16807 * (s - k * 127773) - 2836 * k;
    if (s < 0)
        s += 2147483647;
    seed = uint32_t(s);
    return double(seed % uint32_t(65536)) / 65536.0;
}

__host__ __device__ int RandomInt(uint32_t& seed, int min, int max)
{
    return (int) (RandomDouble(seed) * max + min);
}*/