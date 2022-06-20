
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_TENSOR_VIEW_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_TENSOR_VIEW_HPP

#include <migraphx/gpu/device/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T, index_int N>
struct hip_tensor_view
{
    using value_type                      = T;
    using hip_index                       = typename hip_shape<N>::hip_index;
    __device__ __host__ hip_tensor_view() = default;
    __host__ hip_tensor_view(tensor_view<T> x) : d(x.data()), s(x.get_shape()) {}
    __host__ hip_tensor_view(T* x, const shape& ss) : d(x), s(ss) {}

    MIGRAPHX_DEVICE_CONSTEXPR const hip_shape<N>& get_shape() const { return s; }

    MIGRAPHX_DEVICE_CONSTEXPR index_int size() const { return s.elements(); }

    MIGRAPHX_DEVICE_CONSTEXPR value_type* data() const { return d; }

    template <class U>
    MIGRAPHX_DEVICE_CONSTEXPR value_type& operator[](U i) const
    {
        return d[s.index(i)];
    }

    MIGRAPHX_DEVICE_CONSTEXPR value_type* begin() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR value_type* end() const { return d + size(); }

    private:
    value_type* d = nullptr;
    hip_shape<N> s{};
};

template <index_int N, class T>
hip_tensor_view<T, N> make_hip_view(const shape& s, T* x)
{
    return {x, s};
}

template <index_int N, class T>
hip_tensor_view<T, N> make_hip_view(tensor_view<T> x)
{
    return {x};
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
