#ifndef MIGRAPHX_GUARD_OPERATORS_DEQUANTIZE_LINEAR_HPP
#define MIGRAPHX_GUARD_OPERATORS_DEQUANTIZE_LINEAR_HPP

#include <array>
#include <migraphx/op/common.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/op/tanh.hpp>
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

struct dequantizelinear
{
    int axis = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "dequantizelinear"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        return {inputs[0].type(), inputs[0].lens()};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
