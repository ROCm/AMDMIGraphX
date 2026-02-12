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
static std::vector<instruction_ref> gru_cell(bool is_forward,
                                             module& m,
                                             instruction_ref ins,
                                             std::vector<instruction_ref> inputs,
                                             int linear_before_reset,
                                             const operation& actv_func1,
                                             const operation& actv_func2)
{
    assert(inputs.size() == 6);
    auto seq      = inputs.at(0);
    auto w        = inputs.at(1);
    auto r        = inputs.at(2);
    auto bias     = inputs.at(3);
    auto seq_lens = inputs.at(4);
    auto ih       = inputs.at(5);

    instruction_ref hidden_states = m.end();
    instruction_ref last_output{};
    migraphx::shape seq_shape = seq->get_shape();
    migraphx::shape r_shape   = r->get_shape();
    long hs                   = r_shape.lens()[2];

    migraphx::shape ss(seq_shape.type(), {seq_shape.lens()[1], r_shape.lens()[2]});
    std::vector<float> data(ss.elements(), 1.0f);
    auto l1 = m.add_literal(migraphx::literal{ss, data});

    // w matrix squeeze to 2-dim and do a transpose
    std::vector<int64_t> perm{1, 0};
    auto sw = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), w);
    auto tw = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), sw);

    // r slide to two part, zr and h
    auto sr  = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), r);
    auto rzr = m.insert_instruction(
        ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2 * hs}}}), sr);
    auto trzr = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), rzr);

    auto rh = m.insert_instruction(
        ins, make_op("slice", {{"axes", {0}}, {"starts", {2 * hs}}, {"ends", {3 * hs}}}), sr);
    auto trh = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), rh);

    // initial states
    auto sih  = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ih);
    size_t bs = ih->get_shape().lens()[1];

    // bias
    instruction_ref bwb{};
    instruction_ref brb_zr{};
    instruction_ref brb_h{};
    if(bias != m.end())
    {
        auto sbias = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), bias);
        auto wb    = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {3 * hs}}}), sbias);
        bwb = m.insert_instruction(
            ins,
            make_op("broadcast", {{"axis", 1}, {"out_lens", {bs, static_cast<size_t>(3 * hs)}}}),
            wb);

        auto rb_zr = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {0}}, {"starts", {3 * hs}}, {"ends", {5 * hs}}}),
            sbias);
        auto rb_h = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {0}}, {"starts", {5 * hs}}, {"ends", {6 * hs}}}),
            sbias);
        brb_zr = m.insert_instruction(
            ins,
            make_op("broadcast", {{"axis", 1}, {"out_lens", {bs, static_cast<size_t>(2 * hs)}}}),
            rb_zr);
        brb_h = m.insert_instruction(
            ins,
            make_op("broadcast", {{"axis", 1}, {"out_lens", {bs, static_cast<size_t>(hs)}}}),
            rb_h);
    }

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

        auto xt_w    = m.insert_instruction(ins, make_op("dot"), xt, tw);
        auto ih1_rzr = m.insert_instruction(ins, make_op("dot"), sih, trzr);
        if(bias != m.end())
        {
            xt_w    = m.insert_instruction(ins, make_op("add"), xt_w, bwb);
            ih1_rzr = m.insert_instruction(ins, make_op("add"), ih1_rzr, brb_zr);
        }

        auto xw_z = m.insert_instruction(
            ins, make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {hs}}}), xt_w);
        auto xw_r = m.insert_instruction(
            ins, make_op("slice", {{"axes", {1}}, {"starts", {hs}}, {"ends", {2 * hs}}}), xt_w);
        auto xw_h = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {1}}, {"starts", {2 * hs}}, {"ends", {3 * hs}}}),
            xt_w);

        auto hr_z = m.insert_instruction(
            ins, make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {hs}}}), ih1_rzr);
        auto hr_r = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {1}}, {"starts", {hs}}, {"ends", {2 * hs}}}),
            ih1_rzr);

        auto xw_hr_z = m.insert_instruction(ins, make_op("add"), xw_z, hr_z);
        auto zt      = m.insert_instruction(ins, actv_func1, xw_hr_z);

        auto xw_hr_r = m.insert_instruction(ins, make_op("add"), xw_r, hr_r);
        auto rt      = m.insert_instruction(ins, actv_func1, xw_hr_r);

        instruction_ref hr_h{};
        if(linear_before_reset == 0)
        {
            auto rt_ht1 = m.insert_instruction(ins, make_op("mul"), rt, sih);
            hr_h        = m.insert_instruction(ins, make_op("dot"), rt_ht1, trh);
            if(bias != m.end())
            {
                hr_h = m.insert_instruction(ins, make_op("add"), hr_h, brb_h);
            }
        }
        else
        {
            auto ht1_rh = m.insert_instruction(ins, make_op("dot"), sih, trh);
            if(bias != m.end())
            {
                ht1_rh = m.insert_instruction(ins, make_op("add"), ht1_rh, brb_h);
            }
            hr_h = m.insert_instruction(ins, make_op("mul"), rt, ht1_rh);
        }

        auto xw_hr_h = m.insert_instruction(ins, make_op("add"), xw_h, hr_h);
        auto ht      = m.insert_instruction(ins, actv_func2, xw_hr_h);

        // equation Ht = (1 - zt) (.) ht + zt (.) Ht-1
        auto one_minus_zt    = m.insert_instruction(ins, make_op("sub"), l1, zt);
        auto one_minus_zt_ht = m.insert_instruction(ins, make_op("mul"), one_minus_zt, ht);
        auto zt_ht1          = m.insert_instruction(ins, make_op("mul"), zt, sih);
        sih                  = m.insert_instruction(ins, make_op("add"), one_minus_zt_ht, zt_ht1);
        last_output = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0, 1}}}), sih);

        if(i < seq_len - 1)
        {
            if(is_forward)
            {
                hidden_states =
                    (seq_index == 0)
                        ? last_output
                        : m.insert_instruction(
                              ins, make_op("concat", {{"axis", 0}}), hidden_states, last_output);
            }
            else
            {
                hidden_states =
                    (seq_index == seq_len - 1)
                        ? last_output
                        : m.insert_instruction(
                              ins, make_op("concat", {{"axis", 0}}), last_output, hidden_states);
            }
        }
    }

    return {hidden_states, last_output};
}

static std::vector<operation> get_gru_actv_funcs(const std::vector<operation>& actv_funcs,
                                                 op::rnn_direction direction)
{
    if(direction == op::rnn_direction::bidirectional)
    {
        if(actv_funcs.empty())
            return {make_op("sigmoid"), make_op("tanh"), make_op("sigmoid"), make_op("tanh")};
        else if(actv_funcs.size() == 1)
            return {actv_funcs.at(0), actv_funcs.at(0), actv_funcs.at(0), actv_funcs.at(0)};
        else if(actv_funcs.size() == 2)
            return {actv_funcs.at(0), actv_funcs.at(1), actv_funcs.at(0), actv_funcs.at(1)};
        else if(actv_funcs.size() == 3)
            return {actv_funcs.at(0), actv_funcs.at(1), actv_funcs.at(2), actv_funcs.at(0)};
        else
            return actv_funcs;
    }
    else
    {
        if(actv_funcs.empty())
            return {make_op("sigmoid"), make_op("tanh")};
        else if(actv_funcs.size() == 1)
            return {actv_funcs.at(0), actv_funcs.at(0)};
        else
            return actv_funcs;
    }
}

struct gru_builder : op_builder<gru_builder>
{
    static std::vector<std::string> names() { return {"gru"}; }

    std::size_t hidden_size     = 1;
    std::vector<operation> actv_funcs{};
    op::rnn_direction direction = op::rnn_direction::forward;
    float clip                  = 0.0f;
    int linear_before_reset     = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.hidden_size, "hidden_size"),
                    f(self.actv_funcs, "actv_func"),
                    f(self.direction, "direction"),
                    f(self.clip, "clip"),
                    f(self.linear_before_reset, "linear_before_reset"));
    }

    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto resolved_actv = get_gru_actv_funcs(actv_funcs, direction);

        shape seq_shape         = args[0]->get_shape();
        std::size_t hs          = args[2]->get_shape().lens()[2];
        std::size_t batch_size  = seq_shape.lens()[1];
        shape::type_t type      = seq_shape.type();
        migraphx::shape ih_shape{type, {1, batch_size, hs}};
        std::vector<float> data(ih_shape.elements(), 0.0);

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
            // w weight matrix
            auto w_forward = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                args[1]);
            auto w_reverse = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
                args[1]);

            // r weight matrix
            auto r_forward = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                args[2]);
            auto r_reverse = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
                args[2]);

            // bias
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

            // initial hidden state
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

            auto ret_forward =
                gru_cell(true,
                         m,
                         ins,
                         {local_args[0], w_forward, r_forward, bias_forward, seq_lens, ih_forward},
                         linear_before_reset,
                         resolved_actv.at(0),
                         resolved_actv.at(1));

            if(variable_seq_len)
            {
                local_args[0] = m.insert_instruction(
                    ins, make_op("rnn_var_sl_shift_sequence"), local_args[0], seq_lens);
            }

            auto ret_reverse =
                gru_cell(false,
                         m,
                         ins,
                         {local_args[0], w_reverse, r_reverse, bias_reverse, seq_lens, ih_reverse},
                         linear_before_reset,
                         resolved_actv.at(2),
                         resolved_actv.at(3));

            auto concat_output = m.insert_instruction(
                ins, make_op("concat", {{"axis", 1}}), ret_forward[1], ret_reverse[1]);
            last_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), concat_output);

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

            instruction_ref ih{};
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

            auto ret = gru_cell(is_forward,
                                m,
                                ins,
                                {local_args[0], w, r, bias, seq_lens, ih},
                                linear_before_reset,
                                resolved_actv.at(0),
                                resolved_actv.at(1));

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
