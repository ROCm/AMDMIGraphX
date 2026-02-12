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
static std::vector<instruction_ref> lstm_cell(bool is_forward,
                                              module& m,
                                              instruction_ref ins,
                                              std::vector<instruction_ref> inputs,
                                              const operation& actv_func1,
                                              const operation& actv_func2,
                                              const operation& actv_func3)
{
    assert(inputs.size() == 8);
    auto seq      = inputs.at(0);
    auto w        = inputs.at(1);
    auto r        = inputs.at(2);
    auto bias     = inputs.at(3);
    auto seq_lens = inputs.at(4);
    auto ih       = inputs.at(5);
    auto ic       = inputs.at(6);
    auto pph      = inputs.at(7);

    instruction_ref hidden_states = m.end();
    instruction_ref cell_outputs  = m.end();

    instruction_ref last_hs_output{};
    instruction_ref last_cell_output{};

    migraphx::shape r_shape = r->get_shape();
    long hs                 = r_shape.lens()[2];
    auto bs                 = ih->get_shape().lens()[1];

    std::vector<int64_t> perm{1, 0};
    // w matrix, squeeze and transpose
    auto sw  = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), w);
    auto tsw = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), sw);

    // r matrix, squeeze and transpose
    auto sr  = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), r);
    auto tsr = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), sr);

    // initial hidden state
    auto sih = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ih);

    // initial cell state
    auto sic     = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ic);
    auto ic_lens = sic->get_shape().lens();

    // bias
    instruction_ref wrb{};
    if(bias != m.end())
    {
        auto sbias = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), bias);
        auto ub_wb = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4 * hs}}}), sbias);
        auto ub_rb = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {0}}, {"starts", {4 * hs}}, {"ends", {8 * hs}}}),
            sbias);
        auto ub_wrb = m.insert_instruction(ins, make_op("add"), ub_wb, ub_rb);

        wrb = m.insert_instruction(
            ins,
            make_op("broadcast", {{"axis", 1}, {"out_lens", {bs, 4 * static_cast<size_t>(hs)}}}),
            ub_wrb);
    }

    // peep hole
    instruction_ref pphi_brcst{};
    instruction_ref ppho_brcst{};
    instruction_ref pphf_brcst{};
    if(pph != m.end())
    {
        auto spph = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), pph);
        auto pphi = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {hs}}}), spph);
        pphi_brcst = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", ic_lens}}), pphi);

        auto ppho = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {hs}}, {"ends", {2 * hs}}}), spph);
        ppho_brcst = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", ic_lens}}), ppho);

        auto pphf = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {2 * hs}}, {"ends", {3 * hs}}}), spph);
        pphf_brcst = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", ic_lens}}), pphf);
    }

    long seq_len = rnn_utils::get_seq_len(m, seq, seq_lens);
    for(long i = 0; i < seq_len; ++i)
    {
        long seq_index = is_forward ? i : (seq_len - 1 - i);
        auto xt        = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {0}}, {"starts", {seq_index}}, {"ends", {seq_index + 1}}}),
            seq);
        auto cont_xt = m.insert_instruction(ins, make_op("contiguous"), xt);
        xt           = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), cont_xt);

        auto xt_tsw  = m.insert_instruction(ins, make_op("dot"), xt, tsw);
        auto sih_tsr = m.insert_instruction(ins, make_op("dot"), sih, tsr);
        auto xt_sih  = m.insert_instruction(ins, make_op("add"), xt_tsw, sih_tsr);
        if(bias != m.end())
        {
            xt_sih = m.insert_instruction(ins, make_op("add"), xt_sih, wrb);
        }

        auto it_before_actv = m.insert_instruction(
            ins, make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {hs}}}), xt_sih);
        auto ot_before_actv = m.insert_instruction(
            ins, make_op("slice", {{"axes", {1}}, {"starts", {hs}}, {"ends", {2 * hs}}}), xt_sih);
        auto ft_before_actv = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {1}}, {"starts", {2 * hs}}, {"ends", {3 * hs}}}),
            xt_sih);
        auto ct_before_actv = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {1}}, {"starts", {3 * hs}}, {"ends", {4 * hs}}}),
            xt_sih);

        if(pph != m.end())
        {
            auto pphi_ct   = m.insert_instruction(ins, make_op("mul"), pphi_brcst, sic);
            it_before_actv = m.insert_instruction(ins, make_op("add"), it_before_actv, pphi_ct);

            auto pphf_ct   = m.insert_instruction(ins, make_op("mul"), pphf_brcst, sic);
            ft_before_actv = m.insert_instruction(ins, make_op("add"), ft_before_actv, pphf_ct);
        }
        auto it = m.insert_instruction(ins, actv_func1, it_before_actv);
        auto ft = m.insert_instruction(ins, actv_func1, ft_before_actv);
        auto ct = m.insert_instruction(ins, actv_func2, ct_before_actv);

        // equation Ct = ft (.) Ct-1 + it (.) ct
        auto ft_cell = m.insert_instruction(ins, make_op("mul"), ft, sic);
        auto it_ct   = m.insert_instruction(ins, make_op("mul"), it, ct);
        auto cellt   = m.insert_instruction(ins, make_op("add"), ft_cell, it_ct);

        if(pph != m.end())
        {
            auto ppho_cellt = m.insert_instruction(ins, make_op("mul"), ppho_brcst, cellt);
            ot_before_actv  = m.insert_instruction(ins, make_op("add"), ot_before_actv, ppho_cellt);
        }
        auto ot = m.insert_instruction(ins, actv_func1, ot_before_actv);

        // Ht = ot (.) h(Ct)
        auto h_cellt = m.insert_instruction(ins, actv_func3, cellt);
        auto ht      = m.insert_instruction(ins, make_op("mul"), ot, h_cellt);

        sic = cellt;
        sih = ht;

        last_hs_output = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0, 1}}}), ht);
        last_cell_output =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0, 1}}}), cellt);

        if(i < seq_len - 1)
        {
            if(i == 0)
            {
                hidden_states = last_hs_output;
                cell_outputs  = last_cell_output;
            }
            else
            {
                auto concat_hs_arg0 = is_forward ? hidden_states : last_hs_output;
                auto concat_hs_arg1 = is_forward ? last_hs_output : hidden_states;
                hidden_states       = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), concat_hs_arg0, concat_hs_arg1);

                auto concat_cell_arg0 = is_forward ? cell_outputs : last_cell_output;
                auto concat_cell_arg1 = is_forward ? last_cell_output : cell_outputs;
                cell_outputs          = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), concat_cell_arg0, concat_cell_arg1);
            }
        }
    }

    return {hidden_states, last_hs_output, cell_outputs, last_cell_output};
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static std::vector<operation> get_lstm_actv_funcs(const std::vector<operation>& actv_funcs,
                                                  op::rnn_direction direction)
{
    std::size_t num_actv_funcs = actv_funcs.size();
    if(direction == op::rnn_direction::bidirectional)
    {
        switch(num_actv_funcs)
        {
        case 0:
            return {make_op("sigmoid"),
                    make_op("tanh"),
                    make_op("tanh"),
                    make_op("sigmoid"),
                    make_op("tanh"),
                    make_op("tanh")};

        case 1:
            return {actv_funcs.at(0),
                    actv_funcs.at(0),
                    actv_funcs.at(0),
                    actv_funcs.at(0),
                    actv_funcs.at(0),
                    actv_funcs.at(0)};

        case 2:
            return {actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(1),
                    actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(1)};

        case 3:
            return {actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(2),
                    actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(2)};

        case 4:
            return {actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(2),
                    actv_funcs.at(3),
                    actv_funcs.at(3),
                    actv_funcs.at(3)};

        case 5:
            return {actv_funcs.at(0),
                    actv_funcs.at(1),
                    actv_funcs.at(2),
                    actv_funcs.at(3),
                    actv_funcs.at(4),
                    actv_funcs.at(4)};

        default: return actv_funcs;
        }
    }
    else
    {
        switch(num_actv_funcs)
        {
        case 0: return {make_op("sigmoid"), make_op("tanh"), make_op("tanh")};

        case 1: return {actv_funcs.at(0), actv_funcs.at(0), actv_funcs.at(0)};

        case 2: return {actv_funcs.at(0), actv_funcs.at(1), actv_funcs.at(1)};

        default: return actv_funcs;
        }
    }
}

struct lstm_builder : op_builder<lstm_builder>
{
    static std::vector<std::string> names() { return {"lstm"}; }

    std::vector<operation> actv_funcs{};
    op::rnn_direction direction = op::rnn_direction::forward;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.actv_funcs, "actv_func"),
                    f(self.direction, "direction"));
    }

    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto resolved_actv = get_lstm_actv_funcs(actv_funcs, direction);

        shape seq_shape        = args[0]->get_shape();
        std::size_t hs         = args[2]->get_shape().lens()[2];
        std::size_t batch_size = seq_shape.lens()[1];
        shape::type_t type     = seq_shape.type();
        migraphx::shape ihc_shape{type, {1, batch_size, hs}};
        std::vector<float> ihc_data(ihc_shape.elements(), 0.0);

        // process sequence length
        instruction_ref seq_lens = m.end();
        if((args.size() >= 5) and not args[4]->is_undefined())
        {
            seq_lens = args[4];
        }

        bool variable_seq_len = rnn_utils::is_variable_seq_lens(m, seq_lens);
        auto local_args       = args;

        instruction_ref hidden_states{};
        instruction_ref last_hs_output{};
        instruction_ref cell_outputs{};
        instruction_ref last_cell_output{};
        if(direction == op::rnn_direction::bidirectional)
        {
            // input weight matrix
            auto w_forward = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), args[1]);
            auto w_reverse = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), args[1]);

            // hidden state weight matrix
            auto r_forward = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), args[2]);
            auto r_reverse = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), args[2]);

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
            if(args.size() >= 6 and not args[5]->is_undefined())
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
                ih_forward = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
                ih_reverse = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
            }

            // process initial cell value
            instruction_ref ic_forward{};
            instruction_ref ic_reverse{};
            if(args.size() >= 7 and not args[6]->is_undefined())
            {
                ic_forward = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                    args[6]);
                ic_reverse = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
                    args[6]);
            }
            else
            {
                ic_forward = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
                ic_reverse = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
            }

            // process weight of the peephole
            instruction_ref pph_forward = m.end();
            instruction_ref pph_reverse = m.end();
            if(args.size() == 8 and not args[7]->is_undefined())
            {
                pph_forward = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
                    args[7]);
                pph_reverse = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
                    args[7]);
            }

            auto ret_forward = lstm_cell(true,
                                         m,
                                         ins,
                                         {local_args[0],
                                          w_forward,
                                          r_forward,
                                          bias_forward,
                                          seq_lens,
                                          ih_forward,
                                          ic_forward,
                                          pph_forward},
                                         resolved_actv.at(0),
                                         resolved_actv.at(1),
                                         resolved_actv.at(2));

            if(variable_seq_len)
            {
                local_args[0] = m.insert_instruction(
                    ins, make_op("rnn_var_sl_shift_sequence"), local_args[0], seq_lens);
            }
            auto ret_reverse = lstm_cell(false,
                                         m,
                                         ins,
                                         {local_args[0],
                                          w_reverse,
                                          r_reverse,
                                          bias_reverse,
                                          seq_lens,
                                          ih_reverse,
                                          ic_reverse,
                                          pph_reverse},
                                         resolved_actv.at(3),
                                         resolved_actv.at(4),
                                         resolved_actv.at(5));

            auto concat_hs_output = m.insert_instruction(
                ins, make_op("concat", {{"axis", 1}}), ret_forward[1], ret_reverse[1]);
            auto concat_cell_output = m.insert_instruction(
                ins, make_op("concat", {{"axis", 1}}), ret_forward[3], ret_reverse[3]);
            last_hs_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), concat_hs_output);
            last_cell_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), concat_cell_output);

            if(ret_forward[0] == m.end())
            {
                hidden_states = concat_hs_output;
                cell_outputs  = concat_cell_output;
            }
            else
            {
                ret_forward[1] = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), ret_forward[0], ret_forward[1]);
                ret_reverse[1] = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), ret_reverse[1], ret_reverse[0]);

                ret_forward[3] = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), ret_forward[2], ret_forward[3]);
                ret_reverse[3] = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), ret_reverse[3], ret_reverse[2]);
                cell_outputs = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 1}}), ret_forward[3], ret_reverse[3]);
                hidden_states = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 1}}), ret_forward[1], ret_reverse[1]);
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
            if(args.size() >= 6 and not args[5]->is_undefined())
            {
                ih = args[5];
            }
            else
            {
                ih = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
            }

            instruction_ref ic{};
            if(args.size() >= 7 and not args[6]->is_undefined())
            {
                ic = args[6];
            }
            else
            {
                ic = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
            }

            instruction_ref pph = m.end();
            if(args.size() == 8 and not args[7]->is_undefined())
            {
                pph = args[7];
            }

            if(not is_forward and variable_seq_len)
            {
                local_args[0] = m.insert_instruction(
                    ins, make_op("rnn_var_sl_shift_sequence"), local_args[0], seq_lens);
            }
            auto ret = lstm_cell(is_forward,
                                 m,
                                 ins,
                                 {local_args[0], w, r, bias, seq_lens, ih, ic, pph},
                                 resolved_actv.at(0),
                                 resolved_actv.at(1),
                                 resolved_actv.at(2));

            last_hs_output = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ret[1]);
            last_cell_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ret[3]);

            if(ret[0] == m.end())
            {
                cell_outputs  = ret[3];
                hidden_states = m.insert_instruction(ins, make_op("concat", {{"axis", 0}}), ret[1]);
            }
            else
            {
                auto concat_cell_arg0 = is_forward ? ret[2] : ret[3];
                auto concat_cell_arg1 = is_forward ? ret[3] : ret[2];
                cell_outputs          = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), concat_cell_arg0, concat_cell_arg1);

                auto concat_arg0 = is_forward ? ret[0] : ret[1];
                auto concat_arg1 = is_forward ? ret[1] : ret[0];
                hidden_states    = m.insert_instruction(
                    ins, make_op("concat", {{"axis", 0}}), concat_arg0, concat_arg1);
            }
        }

        // pad hidden states if needed
        hidden_states = rnn_utils::pad_hidden_states(m, ins, args[0], seq_lens, hidden_states);

        // handle variable sequence lengths
        if(variable_seq_len)
        {
            hidden_states =
                rnn_utils::apply_var_sl_shift_hs(m, ins, hidden_states, seq_lens, direction);
            last_hs_output =
                m.insert_instruction(ins,
                                     make_op("rnn_var_sl_last_output", {{"direction", direction}}),
                                     hidden_states,
                                     seq_lens);
            last_cell_output = rnn_utils::compute_var_sl_last_cell_output(
                m, ins, cell_outputs, seq_lens, direction);
        }

        return {hidden_states, last_hs_output, last_cell_output};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
