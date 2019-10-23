#ifndef MIGRAPHX_GUARD_RTGLIB_MULTI_INDEX_HPP
#define MIGRAPHX_GUARD_RTGLIB_MULTI_INDEX_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <index_int N>
struct multi_index
{
    using hip_index = hip_array<index_int, N>;
    hip_index id{};
    hip_index stride{};

    template <class F>
    MIGRAPHX_DEVICE_CONSTEXPR void for_stride(hip_index n, F f) const
    {
        for(hip_index i = id; i < n; i = n.carry(i + stride))
        {
            f(i);
        }
    }
};

template <index_int N>
MIGRAPHX_DEVICE_CONSTEXPR multi_index<N>
make_multi_index(const hip_shape<N>& s, index_int i, index_int n)
{
    return {s.multi(i), s.multi(n)};
}

template <index_int N>
MIGRAPHX_DEVICE_CONSTEXPR multi_index<N>
make_multi_index(const hip_shape<N>& s, index_int i, const hip_array<index_int, N>& n)
{
    return {s.multi(i), n};
}

template <index_int N>
inline auto mi_launch(hipStream_t stream, const hip_shape<N>& s, index_int local = 1024)
{
    assert(s.standard);
    assert(s.elements() > 0);
    index_int n       = s.elements();
    index_int groups  = (n + local - 1) / local;
    index_int nglobal = std::min<index_int>(128, groups) * local;

    assert(groups > 0);
    assert(nglobal > 0);
    auto nglobal_multi = s.multi(nglobal);
    // Skip checking this, since this will cause metadata to not be generated
    // for some unknown reason.
    //
    // assert(std::any_of(nglobal_multi.begin(), nglobal_multi.end(), [](auto x){return x>0;}));

    return [=](auto f) {
        launch(stream, nglobal, local)([=](auto idx) {
            auto midx = make_multi_index(s, idx.global, nglobal_multi);
            midx.for_stride(s.lens, [&](auto i) { f(i); });
        });
    };
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
