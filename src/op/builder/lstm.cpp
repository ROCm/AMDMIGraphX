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
#include <migraphx/program.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/op/builder/lstm.hpp>
#include <migraphx/op/builder/rnn_utils.hpp>
#include <atomic>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {
namespace {

// Unrolling past this length builds a graph that fusion cannot finish in
// reasonable time (CMZ segmentation is 592). Emit loop instead.
constexpr long lstm_loop_seq_threshold = 32;

struct lstm_weights
{
    instruction_ref tsw        = {};
    instruction_ref tsr        = {};
    instruction_ref wrb        = {};
    instruction_ref pphi_brcst = {};
    instruction_ref ppho_brcst = {};
    instruction_ref pphf_brcst = {};
    long hs                    = 0;
    bool has_bias              = false;
    bool has_pph               = false;
};

lstm_weights prepare_lstm_weights(module& m,
                                  instruction_ref ins,
                                  instruction_ref w,
                                  instruction_ref r,
                                  instruction_ref bias,
                                  instruction_ref ih,
                                  instruction_ref pph)
{
    lstm_weights prep;
    migraphx::shape r_shape = r->get_shape();
    prep.hs                 = static_cast<long>(r_shape.lens()[2]);
    auto bs                 = ih->get_shape().lens()[1];
    auto ic_lens            = std::vector<std::size_t>{bs, static_cast<std::size_t>(prep.hs)};

    std::vector<int64_t> perm{1, 0};
    auto sw  = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), w);
    auto tsw = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), sw);
    // loop requires standard inputs; keep the unroll path contiguous as well.
    prep.tsw = m.insert_instruction(ins, make_op("contiguous"), tsw);
    auto sr  = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), r);
    auto tsr = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), sr);
    prep.tsr = m.insert_instruction(ins, make_op("contiguous"), tsr);

    if(bias != m.end())
    {
        prep.has_bias = true;
        auto sbias    = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), bias);
        auto ub_wb    = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {4 * prep.hs}}}),
            sbias);
        auto ub_rb = m.insert_instruction(
            ins,
            make_op("slice",
                    {{"axes", {0}}, {"starts", {4 * prep.hs}}, {"ends", {8 * prep.hs}}}),
            sbias);
        auto ub_wrb = m.insert_instruction(ins, make_op("add"), ub_wb, ub_rb);
        auto wrb    = m.insert_instruction(
            ins,
            make_op("broadcast",
                    {{"axis", 1}, {"out_lens", {bs, 4 * static_cast<std::size_t>(prep.hs)}}}),
            ub_wrb);
        prep.wrb = m.insert_instruction(ins, make_op("contiguous"), wrb);
    }

    if(pph != m.end())
    {
        prep.has_pph = true;
        auto spph    = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), pph);
        auto pphi    = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {prep.hs}}}), spph);
        prep.pphi_brcst = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", ic_lens}}), pphi);
        auto ppho = m.insert_instruction(
            ins,
            make_op("slice", {{"axes", {0}}, {"starts", {prep.hs}}, {"ends", {2 * prep.hs}}}),
            spph);
        prep.ppho_brcst = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", ic_lens}}), ppho);
        auto pphf = m.insert_instruction(
            ins,
            make_op("slice",
                    {{"axes", {0}}, {"starts", {2 * prep.hs}}, {"ends", {3 * prep.hs}}}),
            spph);
        prep.pphf_brcst = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", 1}, {"out_lens", ic_lens}}), pphf);
    }
    return prep;
}

// One timestep. xt, sih, and sic are [batch, hidden]. Returns {ht, cellt}.
std::pair<instruction_ref, instruction_ref> lstm_step(module& m,
                                                      instruction_ref ins,
                                                      instruction_ref xt,
                                                      instruction_ref sih,
                                                      instruction_ref sic,
                                                      const lstm_weights& prep,
                                                      const operation& actv_func1,
                                                      const operation& actv_func2,
                                                      const operation& actv_func3)
{
    auto hs      = prep.hs;
    auto xt_tsw  = m.insert_instruction(ins, make_op("dot"), xt, prep.tsw);
    auto sih_tsr = m.insert_instruction(ins, make_op("dot"), sih, prep.tsr);
    auto xt_sih  = m.insert_instruction(ins, make_op("add"), xt_tsw, sih_tsr);
    if(prep.has_bias)
        xt_sih = m.insert_instruction(ins, make_op("add"), xt_sih, prep.wrb);

    auto it_before_actv = m.insert_instruction(
        ins, make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {hs}}}), xt_sih);
    auto ot_before_actv = m.insert_instruction(
        ins, make_op("slice", {{"axes", {1}}, {"starts", {hs}}, {"ends", {2 * hs}}}), xt_sih);
    auto ft_before_actv = m.insert_instruction(
        ins, make_op("slice", {{"axes", {1}}, {"starts", {2 * hs}}, {"ends", {3 * hs}}}), xt_sih);
    auto ct_before_actv = m.insert_instruction(
        ins, make_op("slice", {{"axes", {1}}, {"starts", {3 * hs}}, {"ends", {4 * hs}}}), xt_sih);

    if(prep.has_pph)
    {
        auto pphi_ct   = m.insert_instruction(ins, make_op("mul"), prep.pphi_brcst, sic);
        it_before_actv = m.insert_instruction(ins, make_op("add"), it_before_actv, pphi_ct);
        auto pphf_ct   = m.insert_instruction(ins, make_op("mul"), prep.pphf_brcst, sic);
        ft_before_actv = m.insert_instruction(ins, make_op("add"), ft_before_actv, pphf_ct);
    }
    auto it = m.insert_instruction(ins, actv_func1, it_before_actv);
    auto ft = m.insert_instruction(ins, actv_func1, ft_before_actv);
    auto ct = m.insert_instruction(ins, actv_func2, ct_before_actv);

    auto ft_cell = m.insert_instruction(ins, make_op("mul"), ft, sic);
    auto it_ct   = m.insert_instruction(ins, make_op("mul"), it, ct);
    auto cellt   = m.insert_instruction(ins, make_op("add"), ft_cell, it_ct);

    if(prep.has_pph)
    {
        auto ppho_cellt = m.insert_instruction(ins, make_op("mul"), prep.ppho_brcst, cellt);
        ot_before_actv  = m.insert_instruction(ins, make_op("add"), ot_before_actv, ppho_cellt);
    }
    auto ot      = m.insert_instruction(ins, actv_func1, ot_before_actv);
    auto h_cellt = m.insert_instruction(ins, actv_func3, cellt);
    auto ht      = m.insert_instruction(ins, make_op("mul"), ot, h_cellt);
    return {ht, cellt};
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::vector<instruction_ref> lstm_cell_unroll(bool is_forward,
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

    auto prep = prepare_lstm_weights(m, ins, w, r, bias, ih, pph);
    auto sih  = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ih);
    auto sic  = m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ic);

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

        auto [ht, cellt] =
            lstm_step(m, ins, xt, sih, sic, prep, actv_func1, actv_func2, actv_func3);
        sic              = cellt;
        sih              = ht;

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

std::string next_lstm_loop_name(bool is_forward)
{
    static std::atomic<std::uint64_t> counter{0};
    return std::string("lstm_loop_") + (is_forward ? "fwd_" : "rev_") +
           std::to_string(counter.fetch_add(1));
}

// Returns {Y [seq,1,batch,hs], last_h [1,1,batch,hs], last_c [1,1,batch,hs]}.
// Sequence and weights are loop-carried (unchanged) so the body does not need
// cross-module references that break ref evaluation.
std::vector<instruction_ref> lstm_cell_loop(bool is_forward,
                                            program& prog,
                                            module& m,
                                            instruction_ref ins,
                                            std::vector<instruction_ref> inputs,
                                            const operation& actv_func1,
                                            const operation& actv_func2,
                                            const operation& actv_func3)
{
    assert(inputs.size() == 8);
    auto seq  = inputs.at(0);
    auto w    = inputs.at(1);
    auto r    = inputs.at(2);
    auto bias = inputs.at(3);
    auto ih   = inputs.at(5);
    auto ic   = inputs.at(6);
    auto pph  = inputs.at(7);

    long seq_len = static_cast<long>(seq->get_shape().lens()[0]);
    auto prep    = prepare_lstm_weights(m, ins, w, r, bias, ih, pph);
    // Sequence often arrives as a transpose (e.g. NCHW pooling → seq-first);
    // loop requires standard layout on every input.
    seq = m.insert_instruction(ins, make_op("contiguous"), seq);
    if(not ih->get_shape().standard())
        ih = m.insert_instruction(ins, make_op("contiguous"), ih);
    if(not ic->get_shape().standard())
        ic = m.insert_instruction(ins, make_op("contiguous"), ic);

    auto* body = prog.create_module(next_lstm_loop_name(is_forward));
    auto iter  = body->add_parameter("iter", shape{shape::int64_type});
    auto cond  = body->add_parameter("cond", shape{shape::bool_type});
    auto h_in  = body->add_parameter("h", ih->get_shape());
    auto c_in  = body->add_parameter("c", ic->get_shape());
    auto b_seq = body->add_parameter("seq", seq->get_shape());
    auto b_tsw = body->add_parameter("tsw", prep.tsw->get_shape());
    auto b_tsr = body->add_parameter("tsr", prep.tsr->get_shape());
    instruction_ref b_wrb = body->end();
    if(prep.has_bias)
        b_wrb = body->add_parameter("wrb", prep.wrb->get_shape());

    int64_t scan_dir = is_forward ? 0 : 1;
    auto xt          = body->add_instruction(
        make_op("scan_slice", {{"axis", 0}, {"direction", scan_dir}}), b_seq, iter);
    xt       = body->add_instruction(make_op("contiguous"), xt);
    xt       = body->add_instruction(make_op("squeeze", {{"axes", {0}}}), xt);
    auto sih = body->add_instruction(make_op("squeeze", {{"axes", {0}}}), h_in);
    auto sic = body->add_instruction(make_op("squeeze", {{"axes", {0}}}), c_in);

    lstm_weights body_prep = prep;
    body_prep.tsw          = b_tsw;
    body_prep.tsr          = b_tsr;
    if(prep.has_bias)
        body_prep.wrb = b_wrb;

    auto [ht, cellt] =
        lstm_step(*body, body->end(), xt, sih, sic, body_prep, actv_func1, actv_func2, actv_func3);
    auto h_out = body->add_instruction(make_op("unsqueeze", {{"axes", {0}}}), ht);
    auto c_out = body->add_instruction(make_op("unsqueeze", {{"axes", {0}}}), cellt);
    // Loop prepends the trip axis, so [1, batch, hs] becomes [seq, 1, batch, hs].
    auto y_t = body->add_instruction(make_op("unsqueeze", {{"axes", {0}}}), ht);

    std::vector<instruction_ref> rets{cond, h_out, c_out, b_seq, b_tsw, b_tsr};
    if(prep.has_bias)
        rets.push_back(b_wrb);
    rets.push_back(y_t);
    body->add_return(rets);

    auto max_iter = m.add_literal(literal{shape{shape::int64_type}, {seq_len}});
    auto cond_lit = m.add_literal(literal{shape{shape::bool_type}, {true}});

    std::vector<instruction_ref> loop_args{max_iter, cond_lit, ih, ic, seq, prep.tsw, prep.tsr};
    if(prep.has_bias)
        loop_args.push_back(prep.wrb);

    auto loop = m.insert_instruction(
        ins,
        make_op("loop",
                {{"max_iterations", seq_len},
                 {"scan_output_directions", std::vector<int64_t>{scan_dir}}}),
        loop_args,
        {body});

    // Carried outputs: h, c, seq, tsw, tsr, [wrb]. Scan Y is next.
    auto last_h = m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", 0}}), loop);
    auto last_c = m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", 1}}), loop);
    const std::size_t y_index = prep.has_bias ? 6 : 5;
    auto y = m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", y_index}}), loop);
    // Match the unroll cell's last_* layout of [1, 1, batch, hs].
    last_h = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0}}}), last_h);
    last_c = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0}}}), last_c);
    return {y, last_h, last_c};
}

bool can_use_lstm_loop(program* prog,
                       module& m,
                       instruction_ref seq,
                       instruction_ref seq_lens,
                       instruction_ref pph,
                       bool variable_seq_len)
{
    if(prog == nullptr or variable_seq_len or pph != m.end())
        return false;
    if(seq->get_shape().dynamic())
        return false;
    return rnn_utils::get_seq_len(m, seq, seq_lens) >= lstm_loop_seq_threshold;
}

} // namespace

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

static std::vector<instruction_ref>
insert_lstm_impl(program* prog,
                 module& m,
                 instruction_ref ins,
                 const std::vector<instruction_ref>& args,
                 const std::vector<operation>& actv_funcs,
                 op::rnn_direction direction);

struct lstm_builder : op_builder<lstm_builder>
{
    static std::vector<std::string> names() { return {"lstm"}; }

    std::vector<operation> actv_funcs{};
    op::rnn_direction direction = op::rnn_direction::forward;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.actv_funcs, "actv_func"), f(self.direction, "direction"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        return insert_lstm_impl(nullptr, m, ins, args, actv_funcs, direction);
    }
};

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static std::vector<instruction_ref>
insert_lstm_impl(program* prog,
                 module& m,
                 instruction_ref ins,
                 const std::vector<instruction_ref>& args,
                 const std::vector<operation>& actv_funcs,
                 op::rnn_direction direction)
{
    auto resolved_actv = get_lstm_actv_funcs(actv_funcs, direction);

    shape seq_shape        = args[0]->get_shape();
    std::size_t hs         = args[2]->get_shape().lens()[2];
    std::size_t batch_size = seq_shape.lens()[1];
    shape::type_t type     = seq_shape.type();
    migraphx::shape ihc_shape{type, {1, batch_size, hs}};
    std::vector<float> ihc_data(ihc_shape.elements(), 0.0);

    instruction_ref seq_lens = m.end();
    if((args.size() >= 5) and not args[4]->is_undefined())
        seq_lens = args[4];

    bool variable_seq_len = rnn_utils::is_variable_seq_lens(m, seq_lens);
    auto local_args       = args;

    instruction_ref hidden_states{};
    instruction_ref last_hs_output{};
    instruction_ref cell_outputs{};
    instruction_ref last_cell_output{};
    if(direction == op::rnn_direction::bidirectional)
    {
        auto w_forward = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), args[1]);
        auto w_reverse = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), args[1]);
        auto r_forward = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), args[2]);
        auto r_reverse = m.insert_instruction(
            ins, make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), args[2]);

        instruction_ref bias_forward = m.end();
        instruction_ref bias_reverse = m.end();
        if(args.size() >= 4 and not args[3]->is_undefined())
        {
            bias_forward = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), args[3]);
            bias_reverse = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), args[3]);
        }

        instruction_ref ih_forward{};
        instruction_ref ih_reverse{};
        if(args.size() >= 6 and not args[5]->is_undefined())
        {
            ih_forward = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), args[5]);
            ih_reverse = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), args[5]);
        }
        else
        {
            ih_forward = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
            ih_reverse = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
        }

        instruction_ref ic_forward{};
        instruction_ref ic_reverse{};
        if(args.size() >= 7 and not args[6]->is_undefined())
        {
            ic_forward = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), args[6]);
            ic_reverse = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), args[6]);
        }
        else
        {
            ic_forward = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
            ic_reverse = m.add_literal(migraphx::literal{ihc_shape, ihc_data});
        }

        instruction_ref pph_forward = m.end();
        instruction_ref pph_reverse = m.end();
        if(args.size() == 8 and not args[7]->is_undefined())
        {
            pph_forward = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), args[7]);
            pph_reverse = m.insert_instruction(
                ins, make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), args[7]);
        }

        bool use_loop = can_use_lstm_loop(
            prog, m, local_args[0], seq_lens, pph_forward, variable_seq_len) and
                        (pph_reverse == m.end());

        if(use_loop)
        {
            auto ret_forward = lstm_cell_loop(true,
                                              *prog,
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
            auto ret_reverse = lstm_cell_loop(false,
                                              *prog,
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
            hidden_states = m.insert_instruction(
                ins, make_op("concat", {{"axis", 1}}), ret_forward[0], ret_reverse[0]);
            auto concat_hs = m.insert_instruction(
                ins, make_op("concat", {{"axis", 1}}), ret_forward[1], ret_reverse[1]);
            auto concat_cell = m.insert_instruction(
                ins, make_op("concat", {{"axis", 1}}), ret_forward[2], ret_reverse[2]);
            last_hs_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), concat_hs);
            last_cell_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), concat_cell);
        }
        else
        {
            auto ret_forward = lstm_cell_unroll(true,
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
            auto ret_reverse = lstm_cell_unroll(false,
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
    }
    else
    {
        bool is_forward = (direction == op::rnn_direction::forward);
        auto w          = args[1];
        auto r          = args[2];

        instruction_ref bias = m.end();
        if(args.size() >= 4 and not args[3]->is_undefined())
            bias = args[3];

        instruction_ref ih{};
        if(args.size() >= 6 and not args[5]->is_undefined())
            ih = args[5];
        else
            ih = m.add_literal(migraphx::literal{ihc_shape, ihc_data});

        instruction_ref ic{};
        if(args.size() >= 7 and not args[6]->is_undefined())
            ic = args[6];
        else
            ic = m.add_literal(migraphx::literal{ihc_shape, ihc_data});

        instruction_ref pph = m.end();
        if(args.size() == 8 and not args[7]->is_undefined())
            pph = args[7];

        if(not is_forward and variable_seq_len)
        {
            local_args[0] = m.insert_instruction(
                ins, make_op("rnn_var_sl_shift_sequence"), local_args[0], seq_lens);
        }

        if(can_use_lstm_loop(prog, m, local_args[0], seq_lens, pph, variable_seq_len))
        {
            auto ret = lstm_cell_loop(is_forward,
                                      *prog,
                                      m,
                                      ins,
                                      {local_args[0], w, r, bias, seq_lens, ih, ic, pph},
                                      resolved_actv.at(0),
                                      resolved_actv.at(1),
                                      resolved_actv.at(2));
            hidden_states    = ret[0];
            last_hs_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ret[1]);
            last_cell_output =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), ret[2]);
        }
        else
        {
            auto ret = lstm_cell_unroll(is_forward,
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
    }

    hidden_states = rnn_utils::pad_hidden_states(m, ins, args[0], seq_lens, hidden_states);

    if(variable_seq_len)
    {
        hidden_states =
            rnn_utils::apply_var_sl_shift_hs(m, ins, hidden_states, seq_lens, direction);
        last_hs_output =
            m.insert_instruction(ins,
                                 make_op("rnn_var_sl_last_output", {{"direction", direction}}),
                                 hidden_states,
                                 seq_lens);
        last_cell_output =
            rnn_utils::compute_var_sl_last_cell_output(m, ins, cell_outputs, seq_lens, direction);
    }

    return {hidden_states, last_hs_output, last_cell_output};
}

std::vector<instruction_ref> insert_lstm(program& prog,
                                         module& m,
                                         instruction_ref ins,
                                         const std::vector<instruction_ref>& args,
                                         const value& options)
{
    auto actv = from_value<std::vector<operation>>(options.at("actv_func"));
    auto dir  = from_value<op::rnn_direction>(options.at("direction"));
    return insert_lstm_impl(&prog, m, ins, args, actv, dir);
}

std::vector<instruction_ref> add_lstm(program& prog,
                                      module& m,
                                      const std::vector<instruction_ref>& args,
                                      const value& options)
{
    return insert_lstm(prog, m, m.end(), args, options);
}

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
