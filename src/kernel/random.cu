/*
 * Created by Maximiliano Levi on 4/9/2021.
 */

#include "random.h"
#include <random>

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

double* Random::GetEntropy()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    auto random_numbers = new double[kRandomCount];
    for(int i = 0; i < kRandomCount; ++i)
    {
        random_numbers[i] = dist(gen);
    }
    return random_numbers;
}

CUDA_HOST_DEVICE double Random::Double()
{
    auto val = entropy_[seed_ % kRandomCount];
    seed_ = (seed_ * 1337) % kRandomCount;
    return val;
}

CUDA_HOST_DEVICE Vector3 Random::PointOnUnitSphere()
{
    double u1 = Double();
    double u2 = Double();
    double lambda = acos(2.0 * u1 - 1) - PI / 2.0;
    double phi = 2.0 * PI * u2;
    return {std::cos(lambda) * std::cos(phi), std::cos(lambda) * std::sin(phi), std::sin(lambda)};
}

Random Random::New(bool gpu)
{
    Random r;
    r.is_in_gpu_ = gpu;
    r.seed_ = 0;
    r.original_ = true;
    double* entropy = GetEntropy();

    if (r.is_in_gpu_) {
        CUDA_CALL(cudaMalloc(&r.entropy_, sizeof(double) * Random::kRandomCount));
        CUDA_CALL(cudaMemcpy(r.entropy_, entropy, sizeof(double) * Random::kRandomCount, cudaMemcpyHostToDevice));
        delete[] entropy;
    }
    else {
        r.entropy_ = entropy;
    }
    return r;
}

CUDA_HOST_DEVICE Random::~Random()
{
    if (original_) {
        if (is_in_gpu_) {
            CUDA_CALL(cudaFree(entropy_));
        }
        else {
            delete[] entropy_;
        }
    }
}

CUDA_HOST_DEVICE Random Random::Reseed(uint32_t new_seed)
{
    Random r;
    r.original_ = false;
    r.is_in_gpu_ = is_in_gpu_;
    r.seed_ = new_seed;
    r.entropy_ = entropy_;
    return r;
}

CUDA_HOST_DEVICE Random::Random()
{
    original_ = false;
}

CUDA_HOST_DEVICE Random::Random(const Random &random)
{
    original_ = false;
    is_in_gpu_ = random.is_in_gpu_;
    seed_ = random.seed_;
    entropy_ = random.entropy_;
}
