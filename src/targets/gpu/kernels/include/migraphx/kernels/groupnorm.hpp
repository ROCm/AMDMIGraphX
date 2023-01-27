#ifndef GUARD_AMDMIGRAPHX_GROUP_NORM_HPP
#define GUARD_AMDMIGRAPHX_GROUP_NORM_HPP

#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/print.hpp>

namespace migraphx {

template<class Output, class T>
__device__ void groupnorm(Output out, T x0) {
    reduce::block::run<Output>([&](auto out_idx, auto r) {
        constexpr auto relements = r.template elements<T>();
        auto z1 = r.reduce(op::sum{}, 0, op::mean<relements>{})(x0);
        auto z4 = r.reduce(op::sum{}, 0, [&](auto x) {
            auto diff = x - z1;
            return (diff * diff) / vec_type<decltype(diff)>{relements};
        })(x0);
        r.outer([&] {
            out[out_idx] = migraphx::rsqrt(z4 + 1e-12);
        });
    });
}

} // namespace migraphx
#endif // GUARD_AMDMIGRAPHX_GROUP_NORM_HPP
