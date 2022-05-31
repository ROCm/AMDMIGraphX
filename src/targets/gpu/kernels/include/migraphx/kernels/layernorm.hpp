#ifndef MIGRAPHX_GUARD_KERNELS_LAYERNORM_HPP
#define MIGRAPHX_GUARD_KERNELS_LAYERNORM_HPP
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/ops.hpp>

namespace migraphx {

template <index_int Axis, class Input, class Output>
__device__ void layernorm(Input input, Output output)
{
    constexpr auto relements = get_shape_c<reduce::with_axis<Input, Axis>>{}.elements() / get_shape_c<Input>{}.elements();
    reduce::block::run<reduce::with_axis<Input, Axis>>([&](auto, auto r) {
        using value_type = typename Input::type;
        auto mean = [&](auto f) {
            return r.reduce(op::sum{}, 0, f)(input) / value_type{relements};
        };
        // mean(x)
        auto mean_x = mean(op::id{});
        // mean(m ^ 2)
        auto mean_m2 = mean([&](auto x) { 
            auto m = x - mean_x;
            return m * m;
        });

        r.inner([&](auto& y, auto x) {
            auto m = x - mean_x;
            // m * rsqrt(mean(m ^ 2) + 1e-12)
            y = m * rsqrt(mean_m2 + value_type{1e-12});
        })(output, input);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_LAYERNORM_HPP
