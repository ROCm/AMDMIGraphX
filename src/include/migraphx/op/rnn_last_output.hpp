#ifndef MIGRAPHX_GUARD_OPERATORS_RNN_LAST_OUTPUT_HPP
#define MIGRAPHX_GUARD_OPERATORS_RNN_LAST_OUTPUT_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <migraphx/op/common.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct rnn_last_output
{
    rnn_direction direction = rnn_direction::forward;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.direction, "direction"));
    }

    std::string name() const { return "rnn_last_output"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        auto dims = inputs[0].lens();

        // remove the first dimension, remaing are output shape
        dims.erase(dims.begin());
        return {inputs[0].type(), dims};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto out_comp_lens = args[0].get_shape().lens();
        out_comp_lens[0]   = 1;
        shape out_comp_s{output_shape.type(), out_comp_lens};

        visit_all(result, args[0])([&](auto output, auto input) {
            args[1].visit([&](auto seq_lens) {
                par_for(output_shape.elements(), [&](auto i) {
                    auto idx = out_comp_s.multi(i);
                    auto b   = idx[2];
                    if(direction == rnn_direction::reverse or idx[1] == 1)
                    {
                        idx[0] = 0;
                    }
                    else
                    {
                        idx[0] = seq_lens[b] - 1;
                    }
                    output[i] = input(idx.begin(), idx.end());
                });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
