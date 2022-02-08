#ifndef MIGRAPHX_GUARD_KERNELS_SCATTERND_HPP
#define MIGRAPHX_GUARD_KERNELS_SCATTERND_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/dfor.hpp>
#include <migraphx/kernels/basic_ops.hpp>
#include <args.hpp>

namespace migraphx {

template <class T1, class T2>
struct scatternd_settings
{
    T1 is_add{};
    T2 is_mul{};
};

template <class... Ts>
constexpr scatternd_settings<Ts...> make_scatternd_settings(Ts... xs)
{
    return {xs...};
}

template <class T, class U, class V, class W, class Settings>
__device__ void scatternd(const T& /* data_t */,
                          const U& /* indices_t */,
                          const V& /* updates_t */,
                          const W& out_t,
                          Settings /* s */)
{
    auto index        = make_index();
    auto i            = index.global;
    auto output_shape = out_t.get_shape();

    if(i < output_shape.elements())
    {
        /* const bool is_add = s.is_add;
        const bool is_mul = s.is_mul;
        const auto* data    = data_t.data();
        const auto* indices = indices_t.data();
        const auto* updates  = updates_t.data();
        auto* output_ptr = out_t.data();

        auto updates_shape = updates_t.get_shape();
        auto indices_shape = indices_t.get_shape();
        auto k = indices_shape.lens.back();
        auto r = output_shape.lens.size();
        auto q = indices_shape.lens.size(); */
    }
}

} // namespace migraphx
#endif
