/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/op/builder/rnn_utils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static std::vector<instruction_ref> vanilla_rnn_cell(bool is_forward,
                                                     module& m,
                                                     instruction_ref ins,
                                                     std::vector<instruction_ref> inputs,
                                                     const operation& actv_func)
{
    assert(inputs.size() == 6);
    auto seq      = inputs.at(0);
    auto w        = inputs.at(1);
    auto r        = inputs.at(2);
    auto bias     = inputs.at(3);
    auto seq_lens = inputs.at(4);
    auto ih       = inputs.at(5);

    // squeeze and transpose w
    std::vector<int64_t> perm{1, 0};
    auto sw      = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), w);
    auto tran_sw = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), sw);

    // squeeze and transpose r
    auto sr      = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), r);
    auto tran_sr = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), sr);

    // initial hidden state
    auto sih      = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ih);
    auto sih_lens = sih->get_shape().lens();

    // bias
    instruction_ref bb{};
    if(bias != m.end())
    {
        long hs    = r->get_shape().lens()[2];
        auto sbias = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), bias);
        auto wb    = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {hs}}}), sbias);
        auto rb = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {hs}}, {"ends", {2 * hs}}}), sbias);
        auto wrb = m.insert_instruction(ins, make_op("add"), wb, rb);
        bb       = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", sih_lens}}), wrb);
    }

    instruction_ref hidden_out = m.end();
    instruction_ref last_out{};
    last_out     = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0, 1}}}), sih);
    long seq_len = rnn_utils::get_seq_len(m, seq, seq_lens);
    for(long i = 0; i < seq_len; i++)
    {
        long seq_index = is_forward ? i : (seq_len - 1 - i);
        auto xt        = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {0}}, {"starts", {seq_index}}, {"ends", {seq_index + 1}}}),
            seq);
        auto cont_xt = m.insert_instruction(ins, make_op("contiguous"), xt);
        xt           = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), cont_xt);
        auto xt_wi   = m.insert_instruction(ins, make_op("dot"), xt, tran_sw);
        auto ht_ri   = m.insert_instruction(ins, make_op("dot"), sih, tran_sr);
        if(bias != m.end())
        {
            xt_wi = m.insert_instruction(ins, make_op("add"), xt_wi, bb);
        }
        auto xt_ht = m.insert_instruction(ins, make_op("add"), xt_wi, ht_ri);

        // apply activation function
        auto ht = m.insert_instruction(ins, actv_func, xt_ht);
        sih     = ht;

        last_out = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0, 1}}}), ht);

        if(i < seq_len - 1)
        {
            if(is_forward)
            {
                hidden_out = (seq_index == 0)
                                 ? last_out
                                 : m.insert_instruction(
                                       ins, make_op("concat", {{"axis", 0}}), hidden_out, last_out);
            }
            else
            {
                hidden_out = (seq_index == seq_len - 1)
                                 ? last_out
                                 : m.insert_instruction(
                                       ins, make_op("concat", {{"axis", 0}}), last_out, hidden_out);
            }
        }
    }

    return {hidden_out, last_out};
}

static std::vector<operation> get_vanilla_rnn_actv_funcs(
    const std::vector<operation>& actv_funcs, op::rnn_direction direction)
{
    if(direction == op::rnn_direction::bidirectional)
    {
        if(actv_funcs.empty())
            return {make_op("tanh"), make_op("tanh")};
        else if(actv_funcs.size() == 1)
            return {actv_funcs.at(0), actv_funcs.at(0)};
        else
            return actv_funcs;
    }
    else
    {
        if(actv_funcs.empty())
            return {make_op("tanh")};
        else
            return actv_funcs;
    }
}

struct rnn_builder : op_builder<rnn_builder>
{
    static std::vector<std::string> names() { return {"rnn"}; }

    std::size_t hidden_size = 1;
    std::vector<operation> actv_funcs{};
    op::rnn_direction direction = op::rnn_direction::forward;
    float clip                  = 0.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.hidden_size, "hidden_size"),
                    f(self.actv_funcs, "actv_func"),
                    f(self.direction, "direction"),
                    f(self.clip, "clip"));
    }

    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto resolved_actv = get_vanilla_rnn_actv_funcs(actv_funcs, direction);

        shape seq_shape         = args[0]->get_shape();
        std::size_t hs          = args[1]->get_shape().lens()[1];
        std::size_t batch_size  = seq_shape.lens()[1];
        shape::type_t type      = seq_shape.type();
        migraphx::shape ih_shape{type, {1, batch_size, hs}};
        std::vector<float> data(ih_shape.elements(), 0);

        // process sequence length
        instruction_ref seq_lens = m.end();
        if((args.size() >= 5) and not args[4]->is_undefined())
        {
            seq_lens = args[4];
        }

        bool variable_seq_len = rnn_utils::is_variable_seq_lens(m, seq_lens);
        auto local_args       = args;

        instruction_ref hidden_states{};
        instruction_ref last_output{};
        if(direction == op::rnn_direction::bidirectional)
        {
            // input weight matrix
            auto w_forward = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                args[1]);
            auto w_reverse = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
                args[1]);

            // hidden state weight matrix
            auto r_forward = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                args[2]);
            auto r_reverse = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
                args[2]);

            // process bias
            instruction_ref bias_forward = m.end();
            instruction_ref bias_reverse = m.end();
            if(args.size() >= 4 and not args[3]->is_undefined())
            {
                bias_forward = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                    args[3]);
                bias_reverse = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
                    args[3]);
            }

            // process initial hidden state
            instruction_ref ih_forward{};
            instruction_ref ih_reverse{};
            if(args.size() == 6 and not args[5]->is_undefined())
            {
                ih_forward = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                    args[5]);
                ih_reverse = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
                    args[5]);
            }
            else
            {
                ih_forward = m.add_literal(migraphx::literal{ih_shape, data});
                ih_reverse = m.add_literal(migraphx::literal{ih_shape, data});
            }

            auto ret_forward = vanilla_rnn_cell(
                true,
                m,
                ins,
                {local_args[0], w_forward, r_forward, bias_forward, seq_lens, ih_forward},
                resolved_actv.at(0));

            if(variable_seq_len)
            {
                local_args[0] = m.insert_instruction(
                    ins, make_op("rnn_var_sl_shift_sequence"), local_args[0], seq_lens);
            }

            auto ret_reverse = vanilla_rnn_cell(
                false,
                m,
                ins,
                {local_args[0], w_reverse, r_reverse, bias_reverse, seq_lens, ih_reverse},
                resolved_actv.at(1));

            auto concat_output = m.insert_instruction(
                ins, make_op("concat", {{"axis", 1}}), ret_forward[1], ret_reverse[1]);
            last_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), concat_output);

            // build hidden_states
            if(ret_forward[0] == m.end())
            {
                hidden_states = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 1}}), ret_forward[1], ret_reverse[1]);
            }
            else
            {
                ret_forward[0] = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), ret_forward[0], ret_forward[1]);
                ret_reverse[0] = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), ret_reverse[1], ret_reverse[0]);
                hidden_states = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 1}}), ret_forward[0], ret_reverse[0]);
            }
        }
        else
        {
            bool is_forward = (direction == op::rnn_direction::forward);
            auto w          = args[1];
            auto r          = args[2];

            instruction_ref bias = m.end();
            if(args.size() >= 4 and not args[3]->is_undefined())
            {
                bias = args[3];
            }

            instruction_ref ih;
            if(args.size() == 6 and not args[5]->is_undefined())
            {
                ih = args[5];
            }
            else
            {
                ih = m.add_literal(migraphx::literal{ih_shape, data});
            }

            if(not is_forward and variable_seq_len)
            {
                local_args[0] = m.insert_instruction(
                    ins, make_op("rnn_var_sl_shift_sequence"), local_args[0], seq_lens);
            }

            auto ret = vanilla_rnn_cell(
                is_forward,
                m,
                ins,
                {local_args[0], w, r, bias, seq_lens, ih},
                resolved_actv.at(0));
            last_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ret[1]);

            if(ret[0] == m.end())
            {
                hidden_states =
                    m.insert_instruction(ins, make_op("concat", {{"axis", 0}}), ret[1]);
            }
            else
            {
                auto concat_arg0 = is_forward ? ret[0] : ret[1];
                auto concat_arg1 = is_forward ? ret[1] : ret[0];
                hidden_states    = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), concat_arg0, concat_arg1);
            }
        }

        // pad hidden states if needed
        hidden_states =
            rnn_utils::pad_hidden_states(m, ins, args[0], seq_lens, hidden_states);

        // handle variable sequence lengths for last output
        if(variable_seq_len)
        {
            hidden_states =
                rnn_utils::apply_var_sl_shift_hs(m, ins, hidden_states, seq_lens, direction);
            last_output = m.insert_instruction(
                ins,
                make_op("rnn_var_sl_last_output", {{"direction", direction}}),
                hidden_states,
                seq_lens);
        }

        return {hidden_states, last_output};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
