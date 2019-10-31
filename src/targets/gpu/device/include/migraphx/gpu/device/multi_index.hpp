#ifndef MIGRAPHX_GUARD_RTGLIB_MULTI_INDEX_HPP
#define MIGRAPHX_GUARD_RTGLIB_MULTI_INDEX_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/shape.hpp>
#include <migraphx/functional.hpp>

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

    MIGRAPHX_DEVICE_CONSTEXPR auto for_stride(hip_index n) const
    {
        // f should return void, but this helps with type deduction
        return [=](auto f) -> decltype(f(hip_index{})) {
            for(hip_index i = id; i < n; i = n.carry(i + stride))
            {
                f(i);
            }
        };
    }
};


template<class ForStride>
auto deduce_for_stride(ForStride fs) -> decltype(fs(id{}));

MIGRAPHX_DEVICE_CONSTEXPR multi_index<1>
make_multi_index(index_int i, index_int n)
{
    return {{i}, {n}};
}

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
inline auto mi_nglobal(const hip_shape<N>& s, index_int nlocal)
{
    assert(s.standard);
    assert(s.elements() > 0);
    index_int n       = s.elements();
    index_int groups  = (n + nlocal - 1) / nlocal;
    index_int nglobal = std::min<index_int>(128, groups) * nlocal;

    assert(groups > 0);
    assert(nglobal > 0);
    auto nglobal_multi = s.multi(nglobal);

    // Skip checking this, since this will cause metadata to not be generated
    // for some unknown reason.
    //
    // assert(std::any_of(nglobal_multi.begin(), nglobal_multi.end(), [](auto x){return x>0;}));

    return nglobal_multi;
}

template <index_int N>
inline auto mi_nlocal(const hip_shape<N>& s, index_int local)
{
    assert(s.standard);
    assert(s.elements() > 0);
    auto nlocal_multi = s.multi(local);

    // Skip checking this, since this will cause metadata to not be generated
    // for some unknown reason.
    //
    // assert(std::any_of(nlocal_multi.begin(), nlocal_multi.end(), [](auto x){return x>0;}));

    return nlocal_multi;
}

template <index_int N>
inline auto mi_launch(hipStream_t stream, const hip_shape<N>& global, index_int nlocal = 1024)
{
    auto nglobal_multi = mi_nglobal(global, nlocal);
    auto nglobal       = global.index(nglobal_multi);

    return [=](auto f) {
        launch(stream, nglobal, nlocal)([=](auto idx) {
            auto midx = make_multi_index(global, idx.global, nglobal_multi);
            f(idx, midx.for_stride(global.lens));
        });
    };
}

template <index_int N>
inline auto mi_launch(hipStream_t stream, const hip_shape<N>& global, const hip_shape<N>& local)
{
    auto nlocal        = 1024;
    auto nglobal_multi = mi_nglobal(global, nlocal);
    auto nglobal       = global.index(nglobal_multi);
    auto nlocal_multi  = mi_nlocal(local, nlocal);

    return [=](auto f) {
        launch(stream, nglobal, nlocal)([=](auto idx) {
            auto midx = make_multi_index(global, idx.global, nglobal_multi);
            auto lidx = make_multi_index(local, idx.local, nlocal_multi);
            f(idx, midx.for_stride(global.lens), lidx.for_stride(local.lens));
        });
    };
}

template <index_int N>
inline auto mi_gs_launch(hipStream_t stream, const hip_shape<N>& global, index_int nlocal = 1024)
{
    return [=](auto f) {
        mi_launch(stream, global, nlocal)([=](auto, auto g) { g([&](auto i) { f(i); }); });
    };
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
