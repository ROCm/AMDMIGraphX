#ifndef MIGRAPHX_GUARD_OP_LAYOUT_HPP
#define MIGRAPHX_GUARD_OP_LAYOUT_HPP

#include <migraphx/config.hpp>
#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/op/unary.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct layout : unary<layout>
{
    std::vector<int64_t> permutation;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.permutation, "permutation"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).only_dims(permutation.size());
        auto lens = inputs.at(0).lens();
        auto t    = inputs.at(0).type();
        return shape::from_permutation(t, lens, permutation);
    }

    auto apply() const
    {
        return [](auto x) { return x; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_OP_LAYOUT_HPP
