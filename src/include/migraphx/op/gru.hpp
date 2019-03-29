#ifndef MIGRAPHX_GUARD_OPERATORS_GRU_HPP
#define MIGRAPHX_GUARD_OPERATORS_GRU_HPP

#include <array>
#include <migraphx/op/common.hpp>
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

struct gru
{
    std::size_t hidden_size = 1;
    std::vector<operation> actv_funcs{sigmoid{}, tanh{}};
    rnn_direction direction = rnn_direction::forward;
    float clip              = 0.0f;
    int linear_before_reset = 0;

    std::string name() const { return "gru"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto in_dims     = inputs[0].lens();
        auto hidden_dims = inputs[2].lens();
        if(hidden_size != hidden_dims[2])
        {
            MIGRAPHX_THROW("GRU: hidden size mismatch in attribute and input");
        }

        std::size_t num_directions = 1;
        if(direction == rnn_direction::bidirectional)
        {
            num_directions = 2;
        }

        if(num_directions != hidden_dims[0])
        {
            MIGRAPHX_THROW("GRU: num_direction does not match the direction attribute");
        }

        std::vector<std::size_t> out_dims(in_dims);
        out_dims.insert(out_dims.begin() + 1, num_directions);
        out_dims.back() = hidden_size;

        return {inputs[0].type(), out_dims};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
