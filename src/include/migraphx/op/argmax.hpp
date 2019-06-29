#ifndef MIGRAPHX_GUARD_OPERATORS_ARGMAX_HPP
#define MIGRAPHX_GUARD_OPERATORS_ARGMAX_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct argmax
{
    int axis      = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "argmax"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto lens = inputs[0].lens();
        int n_dim = static_cast<int>(lens.size());
        if(axis >= n_dim || axis < 0)
        {
            MIGRAPHX_THROW("ARGMAX: axis is out of range.");
        }

        lens[axis] = 1;

        return {shape::int64_type, lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
