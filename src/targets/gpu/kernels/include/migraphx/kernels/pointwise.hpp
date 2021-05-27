#ifndef MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP
#define MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP

#include <migraphx/kernels/index.hpp>

namespace migraphx {

template<class F, class T, class... Ts>
__device__ void pointwise(index idx, F f, T out, Ts... xs)
{
    const auto stride = idx.nglobal();
    for(index_int i = idx.global; i < out.get_shape().elements(); i += stride)
    {
        auto multi_idx = out.get_shape().multi(i);
        out[multi_idx] = f(xs[multi_idx]...);
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP

