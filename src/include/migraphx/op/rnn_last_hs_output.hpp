#ifndef MIGRAPHX_GUARD_OPERATORS_RNN_LAST_HS_OUTPUT_HPP
#define MIGRAPHX_GUARD_OPERATORS_RNN_LAST_HS_OUTPUT_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct rnn_last_hs_output
{
    std::string name() const { return "rnn_last_hs_output"; }

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
