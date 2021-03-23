/*
 * Created by Maximiliano Levi on 3/21/2021.
 */

#ifndef RAYTRACER_KERNEL_PTR_H
#define RAYTRACER_KERNEL_PTR_H
#include "../defines.h"

template<class T>
class kernel_ptr {
public:
    CUDA_DEVICE kernel_ptr() : ptr_(nullptr) {};

    CUDA_DEVICE kernel_ptr(T* ptr) : ptr_(ptr) {};

    CUDA_DEVICE kernel_ptr(kernel_ptr<T>&) = delete;

    CUDA_DEVICE kernel_ptr(kernel_ptr<T>&& smart_ptr) noexcept
    {
        ptr_ = smart_ptr.ptr_;
        smart_ptr.ptr_ = nullptr;
    }

    CUDA_DEVICE ~kernel_ptr()
    {
        if (ptr_ != nullptr)
            CUDA_CALL(cudaFree(ptr_));
    }

    CUDA_DEVICE inline kernel_ptr<T>& operator=(kernel_ptr<T>&& smart_ptr) noexcept
    {
        ptr_ = smart_ptr.ptr_;
        smart_ptr.ptr_ = nullptr;
        return *this;
    }

    CUDA_DEVICE inline T* operator->() const
    {
        return ptr_;
    }

private:
    T* ptr_;
};

#endif //RAYTRACER_KERNEL_PTR_H
