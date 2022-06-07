#ifndef MIGRAPHX_GUARD_KERNELS_SOFTMAX_HPP
#define MIGRAPHX_GUARD_KERNELS_SOFTMAX_HPP

#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/ops.hpp>

namespace migraphx {

template <index_int Axis, class Input, class Output>
__device__ void softmax(Input input, Output output)
{
    reduce::block::run<reduce::with_axis<Input, Axis>>([&](auto, auto r) {
        auto batch_max = r.reduce(op::max{}, lowest{}, op::id{})(input);
        auto batch_sum =
            r.reduce(op::sum{}, 0, [&](auto x) { return migraphx::exp(x - batch_max); })(input);
        r.inner([&](auto& y, auto x) { y = migraphx::exp(x - batch_max) / batch_sum; })(output,
                                                                                        input);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_SOFTMAX_HPP
