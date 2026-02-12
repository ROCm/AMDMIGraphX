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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_OP_BUILDER_RNN_UTILS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_OP_BUILDER_RNN_UTILS_HPP

#include <algorithm>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {
namespace rnn_utils {

inline bool is_variable_seq_lens(const module& m, instruction_ref seq_lens)
{
    bool is_var_lens = false;
    if(seq_lens != m.end())
    {
        if(seq_lens->can_eval())
        {
            auto arg_lens = seq_lens->eval();
            std::vector<int64_t> vec_lens;
            arg_lens.visit([&](auto l) { vec_lens.assign(l.begin(), l.end()); });
            int64_t l = 0;
            if(not vec_lens.empty())
            {
                l = vec_lens[0];
            }
            if(not std::all_of(vec_lens.begin(), vec_lens.end(), [&](auto v) { return v == l; }))
            {
                is_var_lens = true;
            }
        }
        else
        {
            is_var_lens = true;
        }
    }

    return is_var_lens;
}

inline std::size_t get_seq_len(const module& m, instruction_ref input, instruction_ref seq_lens)
{
    bool is_var_lens = is_variable_seq_lens(m, seq_lens);
    auto input_shape = input->get_shape();
    auto length      = input_shape.lens()[0];
    if(not is_var_lens and seq_lens != m.end())
    {
        auto arg_len = seq_lens->eval();
        std::vector<std::size_t> vec_lens;
        arg_len.visit([&](auto l) { vec_lens.assign(l.begin(), l.end()); });
        length = vec_lens.empty() ? length : vec_lens[0];
    }

    return length;
}

inline instruction_ref pad_hidden_states(module& m,
                                         instruction_ref ins,
                                         instruction_ref seq,
                                         instruction_ref seq_lens,
                                         instruction_ref hs)
{
    auto max_seq_len = seq->get_shape().lens()[0];
    auto seq_len     = get_seq_len(m, seq, seq_lens);

    auto hs_padded = hs;
    if(seq_len < max_seq_len)
    {
        auto s        = hs->get_shape();
        auto pad_lens = s.lens();
        pad_lens[0]   = static_cast<std::size_t>(max_seq_len - seq_len);
        shape pad_s{s.type(), pad_lens};
        std::vector<float> pad_data(pad_s.elements(), 0.0f);
        auto pl   = m.add_literal(pad_s, pad_data.begin(), pad_data.end());
        hs_padded = m.insert_instruction(ins, make_op("concat", {{"axis", 0}}), hs, pl);
    }

    return hs_padded;
}

inline instruction_ref compute_var_sl_last_hs_output(module& m,
                                                     instruction_ref ins,
                                                     instruction_ref hidden_states,
                                                     instruction_ref seq_lens,
                                                     op::rnn_direction dirct)
{
    auto shifted_hs =
        m.insert_instruction(ins,
                             make_op("rnn_var_sl_shift_output",
                                     {{"output_name", "hidden_states"}, {"direction", dirct}}),
                             hidden_states,
                             seq_lens);
    auto last_output = m.insert_instruction(
        ins, make_op("rnn_var_sl_last_output", {{"direction", dirct}}), shifted_hs, seq_lens);
    return last_output;
}

inline instruction_ref compute_var_sl_last_cell_output(module& m,
                                                       instruction_ref ins,
                                                       instruction_ref cell_outputs,
                                                       instruction_ref seq_lens,
                                                       op::rnn_direction dirct)
{
    auto shifted_co =
        m.insert_instruction(ins,
                             make_op("rnn_var_sl_shift_output",
                                     {{"output_name", "cell_outputs"}, {"direction", dirct}}),
                             cell_outputs,
                             seq_lens);
    auto last_cell = m.insert_instruction(
        ins, make_op("rnn_var_sl_last_output", {{"direction", dirct}}), shifted_co, seq_lens);
    return last_cell;
}

inline instruction_ref apply_var_sl_shift_hs(module& m,
                                             instruction_ref ins,
                                             instruction_ref hidden_states,
                                             instruction_ref seq_lens,
                                             op::rnn_direction dirct)
{
    return m.insert_instruction(
        ins,
        make_op("rnn_var_sl_shift_output",
                {{"output_name", "hidden_states"}, {"direction", dirct}}),
        hidden_states,
        seq_lens);
}

} // namespace rnn_utils
} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
