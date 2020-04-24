#ifndef MIGRAPHX_GUARD_OPERATORS_RNN_CLEAR_MISSING_FRAMES_HPP
#define MIGRAPHX_GUARD_OPERATORS_RNN_CLEAR_MISSING_FRAMES_HPP

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

struct rnn_clear_missing_frames
{
    std::string name() const { return "rnn_clear_missing_frames"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return inputs[0];
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        int64_t max_len = static_cast<int64_t>(output_shape.lens()[0]);
        visit_all(result, args[0])([&](auto output, auto input) {
            using value_type = typename decltype(output)::value_type;
            args[1].visit([&](auto seq_lens) {
                par_for(output_shape.elements(), [&](auto i) {
                    auto idx = output_shape.multi(i);
                    auto batch_id = idx[2];
                    auto d = idx[1];
                    auto t = indx[0];
                    auto sl = seq_lens[batch_id];
                    value_type val = 0;
                    if (t < sl)
                    {
                        auto in_idx = idx;
                        in_idx[0] += d * (max_len - sl);
                        val = input(idx.begin(), idx.end());
                    }
                    output(idx.begin(), idx.end()) = val;
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
