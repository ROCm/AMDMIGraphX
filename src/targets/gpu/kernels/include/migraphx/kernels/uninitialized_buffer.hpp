#ifndef MIGRAPHX_GUARD_KERNELS_UNINITIALIZED_BUFFER_HPP
#define MIGRAPHX_GUARD_KERNELS_UNINITIALIZED_BUFFER_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/debug.hpp>

namespace migraphx {

template <typename T, index_int N>
struct uninitialized_buffer
{
    // Use aligned char storage to avoid initialization
    alignas(T) char storage[sizeof(T) * N];

    __device__ T* data() { return reinterpret_cast<T*>(storage); }

    __device__ const T* data() const { return reinterpret_cast<const T*>(storage); }

    // Array-like interface
    __device__ T& operator[](index_int i)
    {
        MIGRAPHX_ASSERT(i < N);
        return data()[i];
    }

    __device__ const T& operator[](index_int i) const
    {
        MIGRAPHX_ASSERT(i < N);
        return data()[i];
    }

    __device__ constexpr index_int size() const { return N; }

    // Iterator support (if needed)
    __device__ T* begin() { return data(); }
    __device__ T* end() { return data() + N; }
    __device__ const T* begin() const { return data(); }
    __device__ const T* end() const { return data() + N; }
};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_UNINITIALIZED_BUFFER_HPP
