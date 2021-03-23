/*
 * Created by Maximiliano Levi on 3/21/2021.
 */

#ifndef RAYTRACER_KERNEL_VECTOR_H
#define RAYTRACER_KERNEL_VECTOR_H

#include "../defines.h"

template<class T> class kernel_vector {
public:
    CUDA_DEVICE kernel_vector()
    {
        array_ = nullptr;
        count_ = 0;
        array_size_ = 0;
        this->resize(8);
    }

    CUDA_DEVICE ~kernel_vector()
    {
        if (array_ != nullptr)
            CUDA_CALL(cudaFree(array_));
    }

    CUDA_DEVICE void push_back(T &element)
    {
        if (count_ == array_size_)
            this->resize(array_size_ * 2);
        array_[count_++] = element;
    }

    CUDA_DEVICE void push_back(T &&element)
    {
        if (count_ == array_size_)
            this->resize(array_size_ * 2);
        array_[count_++] = static_cast<T&&>(element);
    }

    CUDA_DEVICE inline size_t size() const
    {
        return count_;
    }

    CUDA_DEVICE inline size_t empty() const
    {
        return size() == 0;
    }

    CUDA_DEVICE inline T& operator[](int index)
    {
        return this->array_[index];
    }

    CUDA_DEVICE inline const T& operator[](int index) const
    {
        return this->array_[index];
    }

private:
    CUDA_DEVICE void resize(size_t new_size)
    {
        T* old_ptr = this->array_;
        CUDA_CALL(cudaMalloc(&this->array_, new_size * sizeof(T)));
        if (old_ptr != nullptr) {
            CUDA_CALL(cudaMemcpy(this->array_, old_ptr, count_ * sizeof(T), cudaMemcpyDeviceToDevice));
            CUDA_CALL(cudaFree(old_ptr));
        }
        this->array_size_ = new_size;
    }

    T* array_;
    size_t count_;
    size_t array_size_;
};

#endif //RAYTRACER_KERNEL_VECTOR_H
