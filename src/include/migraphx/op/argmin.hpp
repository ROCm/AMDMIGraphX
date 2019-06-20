#ifndef MIGRAPHX_GUARD_OPERATORS_ARGMIN_HPP
#define MIGRAPHX_GUARD_OPERATORS_ARGMIN_HPP

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

struct argmin
{
    int axis      = 0;
    int keep_dims = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"), f(self.keep_dims, "keep_dims"));
    }

    std::string name() const { return "argmin"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto lens = inputs[0].lens();
        int n_dim = static_cast<int>(lens.size());
        if(axis >= n_dim || axis < 0)
        {
            MIGRAPHX_THROW("ARGMIN: axis is out of range.");
        }

        lens[axis] = 1;
        if(!keep_dims)
        {
            lens.erase(lens.begin() + axis);
        }

        return {shape::int64_type, lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
