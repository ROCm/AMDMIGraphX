#ifndef MIGRAPHX_GUARD_OPERATORS_RNN_VAR_SL_LAST_OUTPUT_HPP
#define MIGRAPHX_GUARD_OPERATORS_RNN_VAR_SL_LAST_OUTPUT_HPP

#include <array>
#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct rnn_var_sl_last_output
{
    rnn_direction direction = rnn_direction::forward;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.direction, "direction"));
    }

    std::string name() const { return "rnn_var_sl_last_output"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
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
