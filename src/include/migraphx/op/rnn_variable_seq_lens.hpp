#ifndef MIGRAPHX_GUARD_OPERATORS_RNN_VARIABLE_SEQ_LENS_HPP
#define MIGRAPHX_GUARD_OPERATORS_RNN_VARIABLE_SEQ_LENS_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct rnn_shift_output
{
    std::string output_name = "hidden_states";
    rnn_direction direction = rnn_direction::forward;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_name, "hidden_states"), f(self.direction, "direction"));
    }

    std::string name() const { return "rnn_shift_output"; }
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
                    auto idx       = output_shape.multi(i);
                    auto batch_id  = idx[2];
                    auto d         = idx[1];
                    auto t         = idx[0];
                    auto sl        = seq_lens[batch_id];
                    value_type val = value_type{0};
                    if(t < sl)
                    {
                        auto in_idx = idx;
                        int offset  = (direction == rnn_direction::reverse or d == 1) ? 1 : 0;
                        in_idx[0] += offset * (max_len - sl);
                        val = input(in_idx.begin(), in_idx.end());
                    }
                    output(idx.begin(), idx.end()) = val;
                });
            });
        });

        return result;
    }
};

struct rnn_shift_sequence
{
    std::string name() const { return "rnn_shift_sequence"; }
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
                    auto idx       = output_shape.multi(i);
                    auto b         = idx[1];
                    auto t         = idx[0];
                    auto sl        = seq_lens[b];
                    value_type val = value_type{0};
                    if(t >= max_len - sl)
                    {
                        auto in_idx = idx;
                        in_idx[0] -= (max_len - sl);
                        val = input(in_idx.begin(), in_idx.end());
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
