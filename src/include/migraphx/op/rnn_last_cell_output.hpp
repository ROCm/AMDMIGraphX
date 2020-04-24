#ifndef MIGRAPHX_GUARD_OPERATORS_RNN_LAST_CELL_OUTPUT_HPP
#define MIGRAPHX_GUARD_OPERATORS_RNN_LAST_CELL_OUTPUT_HPP

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

struct lstm_last_cell_output
{
    std::string name() const { return "lstm_last_cell_output"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        // check_shapes{inputs, *this}.has(1);
        auto dims = inputs[0].lens();

        // remove the first dimension, remaing are output shape
        dims.erase(dims.begin());
        return {inputs[0].type(), dims};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
