
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_VECTOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_VECTOR_HPP

#include <migraphx/gpu/device/types.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T, index_int N>
struct hip_vector
{
    MIGRAPHX_DEVICE_CONSTEXPR hip_vector() = default;
    MIGRAPHX_DEVICE_CONSTEXPR hip_vector(index_int s) : len(s) {}
    template <class Iterator>
    __device__ __host__ hip_vector(Iterator start, Iterator last)
    {
        auto it = std::copy(start, last, d);
        len     = std::distance(d, it);
    }

    __device__ __host__ hip_vector(std::initializer_list<T> x)
    {
        std::copy(x.begin(), x.end(), d);
        len = x.size();
    }

    MIGRAPHX_DEVICE_CONSTEXPR T& operator[](index_int i) { return d[i]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& operator[](index_int i) const { return d[i]; }

    MIGRAPHX_DEVICE_CONSTEXPR T& front() { return d[0]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& front() const { return d[0]; }

    MIGRAPHX_DEVICE_CONSTEXPR T& back() { return d[size() - 1]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& back() const { return d[size() - 1]; }

    MIGRAPHX_DEVICE_CONSTEXPR T* data() { return d; }
    MIGRAPHX_DEVICE_CONSTEXPR const T* data() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR index_int size() const { return len; }

    MIGRAPHX_DEVICE_CONSTEXPR T* begin() { return d; }
    MIGRAPHX_DEVICE_CONSTEXPR const T* begin() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR T* end() { return d + size(); }
    MIGRAPHX_DEVICE_CONSTEXPR const T* end() const { return d + size(); }

    template <class U>
    MIGRAPHX_DEVICE_CONSTEXPR void push_back(U&& x)
    {
        d[len] = static_cast<U&&>(x);
        len++;
    }

    private:
    T d[N]          = {};
    index_int len = 0;
};

template <index_int N, class T>
hip_vector<T, N> to_hip_vector(const std::vector<T>& x)
{
    hip_vector<T, N> result(x.size());
    std::copy(x.begin(), x.end(), result.begin());
    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
