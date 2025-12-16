/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#include <migraphx/fuse_attention.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/match/softmax.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <queue>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_NUM_SPLITS);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_PAGED_ATTN);

std::size_t get_num_splits() { return value_of(MIGRAPHX_FLASH_DECODING_NUM_SPLITS{}, 0); }

// TODO: Write this in matcher.hpp as a general matcher for iterating through inputs
inline auto pointwise_inputs()
{
    return [](auto start, auto f) {
        std::unordered_set<instruction_ref> visited;
        fix([&](auto self, auto ins) {
            if(ins->can_eval())
                return;
            if(not visited.insert(ins).second)
                return;
            if(not ins->get_operator().attributes().contains("pointwise") and
               ins->get_operator().name() != "reshape")
            {
                f(ins);
                return;
            }
            for(auto input : ins->inputs())
                self(input);
        })(start);
    };
}

struct find_attention
{
    std::size_t* counter;

    auto matcher() const
    {
        auto gemm1   = match::any_of[pointwise_inputs()](match::name("dot").bind("dot1"));
        auto softmax = match::skip(match::name("convert"))(match::softmax_input(gemm1));
        return match::name("dot")(match::arg(0)(softmax));
    }

    std::string get_count() const { return std::to_string((*counter)++); }

    std::unordered_map<instruction_ref, instruction_ref>
    invert_map_ins(const std::unordered_map<instruction_ref, instruction_ref>& map_ins) const
    {
        std::unordered_map<instruction_ref, instruction_ref> inverse_map;
        for(auto const& [key, value] : map_ins)
        {
            assert(not contains(inverse_map, value));
            inverse_map[value] = key;
        }
        return inverse_map;
    }

    std::vector<instruction_ref>
    get_attn_instructions(module& m, instruction_ref gemm1, instruction_ref gemm2) const
    {
        auto attn_inss = find_instructions_between(gemm1, gemm2, &m);

        std::vector<instruction_ref> sorted_inss(attn_inss.begin(), attn_inss.end());
        std::sort(
            sorted_inss.begin(), sorted_inss.end(), [&](instruction_ref x, instruction_ref y) {
                return std::distance(m.begin(), x) < std::distance(m.begin(), y);
            });

        return sorted_inss;
    }

    static bool has_lse_out(std::vector<instruction_ref>& group_outs)
    {
        return (group_outs.size() == 3 and
                std::all_of(group_outs.begin(), group_outs.end(), [](auto o) {
                    return contains({"dot", "reduce_max", "reduce_sum"}, o->name());
                }));
    }

    std::vector<instruction_ref>
    get_lse_instructions(std::vector<instruction_ref>& group_outs) const
    {
        std::vector<instruction_ref> lse_inss;
        auto rsum = *std::find_if(
            group_outs.begin(), group_outs.end(), [](auto o) { return o->name() == "reduce_sum"; });
        auto rsum_outs = rsum->outputs();
        auto log       = std::find_if(
            rsum_outs.begin(), rsum_outs.end(), [](auto o) { return o->name() == "log"; });
        if(log == rsum_outs.end())
            return lse_inss;
        auto log_outs = (*log)->outputs();
        if(log_outs.size() != 1 or log_outs.front()->name() != "add")
            return lse_inss;
        auto add = log_outs.front();
        lse_inss.insert(lse_inss.end(), {*log, add});

        return lse_inss;
    }

    std::vector<instruction_ref> find_outputs(std::vector<instruction_ref> inss) const
    {
        std::vector<instruction_ref> outputs;
        std::copy_if(inss.begin(), inss.end(), std::back_inserter(outputs), [&](auto i) {
            return not std::all_of(i->outputs().begin(), i->outputs().end(), [&](auto o) {
                return contains(inss, o);
            });
        });
        return outputs;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto gemm2         = r.result;
        auto gemm1         = r.instructions["dot1"];
        auto softmax_input = r.instructions["x"];

        // Capture all instructions part of the attention op
        auto attn_inss = get_attn_instructions(mpm.get_module(), gemm1, gemm2);

        // Add captured instructions to new submodule
        module m_attn;
        std::unordered_map<instruction_ref, instruction_ref> map_mm_to_mattn;
        m_attn.fuse(attn_inss, &map_mm_to_mattn);

        // Define outputs based on instructions that are used elsewhere in the graph
        auto required_outputs = find_outputs(attn_inss);

        assert(not required_outputs.empty());

        // LSE case requires output from reduce_max and reduce_sum instructions
        if(has_lse_out(required_outputs))
        {
            auto lse_inss = get_lse_instructions(required_outputs);
            m_attn.fuse(lse_inss, &map_mm_to_mattn);

            // Recompute required outputs after adding lse instructions
            attn_inss.insert(attn_inss.end(), lse_inss.begin(), lse_inss.end());
            required_outputs = find_outputs(attn_inss);

            // Final outputs should be the second gemm and add instruction
            // (shift lse to adjust for subtracting max during stable softmax computation)
            if(required_outputs.size() != 2)
                return;
        }
        else if(required_outputs.size() > 1)
        {
            return;
        }

        // Find corresponding output instructions in m_attn
        std::vector<instruction_ref> m_attn_outputs;
        std::transform(required_outputs.begin(),
                       required_outputs.end(),
                       std::back_inserter(m_attn_outputs),
                       [&](auto i) { return map_mm_to_mattn.at(i); });

        m_attn.add_return(m_attn_outputs);

        // Define inputs to m_attn
        auto map_mattn_to_mm = invert_map_ins(map_mm_to_mattn);
        auto new_inputs      = m_attn.get_inputs(map_mattn_to_mm);

        module_ref mpm_attn = mpm.create_module("attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();

        auto group_ins = mpm.get_module().insert_instruction(
            softmax_input, make_op("group", {{"tag", "attention"}}), new_inputs, {mpm_attn});

        if(m_attn_outputs.size() == 1)
        {
            mpm.get_module().replace_instruction(required_outputs.front(), group_ins);
        }
        else
        {
            for(std::size_t i = 0; i < required_outputs.size(); ++i)
            {
                mpm.get_module().replace_instruction(
                    required_outputs[i], make_op("get_tuple_elem", {{"index", i}}), group_ins);
            }
        }
    }
};

struct find_flash_decoding
{
    // number of groups. User-provided for now
    std::size_t groups;

    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "attention")).bind("group");
    }

    std::pair<instruction_ref, instruction_ref> get_gemms(module_ref submod) const
    {
        std::vector<instruction_ref> gemms;
        for(auto it = submod->begin(); it != submod->end(); ++it)
        {
            if(it->name() == "dot")
                gemms.push_back(it);
        }
        assert(gemms.size() == 2 and "Expected exactly 2 gemm operations in attention submodule");

        // gemms[0] is Q@K, gemms[1] is P@V
        // gemms are in order since we iterate from begin to end
        return {gemms[0], gemms[1]};
    }

    std::vector<shape> get_qkv_shapes(instruction_ref q, instruction_ref k, instruction_ref v) const
    {
        std::vector<shape> qkv_shapes;
        auto q_shape = q->get_shape();
        auto k_shape = k->get_shape();
        auto v_shape = v->get_shape();

        qkv_shapes.push_back(q_shape);
        qkv_shapes.push_back(k_shape);
        qkv_shapes.push_back(v_shape);

        assert((q_shape.lens().size() == 3 or q_shape.lens().size() == 4) and
               "Expected 3D or 4D Q, K, V shapes");
        assert(k_shape.lens().size() == q_shape.lens().size() and
               v_shape.lens().size() == q_shape.lens().size() and
               "Q, K, V must have same number of dimensions");
        return qkv_shapes;
    }

    struct transformed_shapes_result
    {
        std::vector<size_t> q_shape;           // final Q shape: [B, G, M, k]
        std::vector<size_t> k_intermediate;    // K intermediate: [B, k, G, N/G]
        std::vector<size_t> k_shape;           // final K shape: [B, G, k, N/G]
        std::vector<int64_t> k_transpose_perm; // permutation for K transpose
        std::vector<size_t> v_shape;           // final V shape: [B, G, N/G, D]
    };

    transformed_shapes_result get_transformed_shapes(const std::vector<shape>& input_shapes) const
    {
        assert(input_shapes.size() == 3 and "Expected Q, K, V shapes");

        auto q_lens = input_shapes[0].lens();
        auto k_lens = input_shapes[1].lens();
        auto v_lens = input_shapes[2].lens();

        // 3D: Q_lens = [B, M, k]
        // 4D: Q_lens = [B, H, M, k]
        size_t ndim = q_lens.size();
        size_t n    = k_lens[ndim - 1];
        size_t g    = groups;

        // TODO: handle uneven splits; this is caught in `apply` for now
        assert(n % g == 0 and
               "Key-value sequence length must be divisible by number of splits/groups");
        size_t n_split = n / g;

        transformed_shapes_result result;

        // batch_dims + G + spatial_dims
        auto insert_g = [&](const auto& lens) {
            std::vector<size_t> new_lens(lens.begin(), lens.begin() + ndim - 2);  // batch dims
            new_lens.push_back(g);                                                // insert G
            new_lens.insert(new_lens.end(), lens.begin() + ndim - 2, lens.end()); // last 2 dims
            return new_lens;
        };

        // Q: [B, M, k] -> [B, G, M, k] via unsqueeze + broadcast
        result.q_shape = insert_g(q_lens);

        // K: [B, k, N] -> [B, G, k, N/G] via reshape + transpose
        // intermediate shape for reshape: [B, k, G, N/G]
        result.k_intermediate.clear();
        for(size_t i = 0; i < k_lens.size() - 1; ++i)
        {
            result.k_intermediate.push_back(k_lens[i]);
        }
        result.k_intermediate.push_back(g);
        result.k_intermediate.push_back(n_split);

        // transpose permutation to get [B, G, k, N/G]
        result.k_transpose_perm.clear();
        for(size_t i = 0; i < k_lens.size() - 2; ++i)
        {
            result.k_transpose_perm.push_back(i); // batch dims stay in place
        }
        result.k_transpose_perm.push_back(k_lens.size() - 1); // G dimension
        result.k_transpose_perm.push_back(k_lens.size() - 2); // k dimension
        result.k_transpose_perm.push_back(k_lens.size());     // N/G dimension

        // final K shape after transpose
        result.k_shape                            = insert_g(k_lens);
        result.k_shape[result.k_shape.size() - 1] = n_split;

        // V: [B, N, D] -> [B, G, N/G, D] via direct reshape
        result.v_shape                            = insert_g(v_lens);
        result.v_shape[result.v_shape.size() - 2] = n_split;

        return result;
    }

    std::unordered_map<instruction_ref, instruction_ref>
    map_submod_params_to_inputs(module_ref submod,
                                const std::vector<instruction_ref>& group_inputs) const
    {
        auto map_param_to_main = submod->get_ins_param_map(group_inputs, true);
        // verify the mapping is correct
        auto expected_inputs = submod->get_inputs(map_param_to_main);
        assert(expected_inputs == group_inputs and "Mapped inputs don't match group inputs");
        return map_param_to_main;
    }

    void rebuild_attention_submodule(
        module& target_mod,
        const module& source_mod,
        const std::unordered_map<instruction_ref, instruction_ref>& param_map) const
    {
        // map from instructions in the old module to the new ones in the target module
        std::unordered_map<instruction_ref, instruction_ref> map_old_to_new = param_map;
        std::unordered_map<std::string, instruction_ref> softmax_parts;

        for(auto it = source_mod.begin(); it != source_mod.end(); ++it)
        {
            auto ins = it;
            if(ins->name() == "@param" or ins->name() == "@return")
                continue;

            // gather inputs for the new instruction
            std::vector<instruction_ref> new_inputs;
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           std::back_inserter(new_inputs),
                           [&](auto i) {
                               assert(contains(map_old_to_new, i) and "Input not found in map");
                               return map_old_to_new.at(i);
                           });

            auto op = ins->get_operator();

            // transform operators that depend on tensor shape/rank
            // adjust reduction axes for the new rank
            if(op.name() == "reduce_max" or op.name() == "reduce_sum")
            {
                auto original_axes = op.to_value()["axes"].to_vector<int64_t>();
                assert(original_axes.size() == 1 and "Expected single axis for reduction");

                const auto& new_input_shape = new_inputs.front()->get_shape();
                assert(original_axes.front() ==
                           static_cast<int64_t>(ins->inputs().front()->get_shape().lens().size() -
                                                1) or
                       original_axes.front() == -1);
                op.from_value(
                    {{"axes", {static_cast<int64_t>(new_input_shape.lens().size() - 1)}}});
            }
            // TODO make less reliant on ops around it
            else if(op.name() == "multibroadcast")
            {
                // broadcast target shape is the shape of the
                // other input to the 'sub' or 'div' instruction.
                auto parent = ins->outputs().front();
                assert(parent->name() == "sub" or parent->name() == "div");

                // Find the sibling input that isn't the reduction result
                auto sibling = std::find_if(parent->inputs().begin(),
                                            parent->inputs().end(),
                                            [&](auto i) { return i != ins; });
                assert(sibling != parent->inputs().end() and
                       "Could not find sibling for broadcast target");

                const auto& target_shape = map_old_to_new.at(*sibling)->get_shape();
                op.from_value({{"out_lens", target_shape.lens()}});
            }

            auto new_ins        = target_mod.add_instruction(op, new_inputs);
            map_old_to_new[ins] = new_ins;

            // store key softmax components for LSE calculation
            if(op.name() == "reduce_max")
                softmax_parts["max"] = new_ins;
            if(op.name() == "reduce_sum")
                softmax_parts["sum_exp"] = new_ins;
        }

        // get the final partial output (O')
        auto orig_return_ins        = std::prev(source_mod.end())->inputs().front();
        auto partial_output_o_prime = map_old_to_new.at(orig_return_ins);

        // calculate LSE = max(S) + log(sum(exp(S - max(S))))
        assert(contains(softmax_parts, "max") and contains(softmax_parts, "sum_exp"));
        auto log_sum_exp = target_mod.add_instruction(make_op("log"), softmax_parts["sum_exp"]);
        auto lse = target_mod.add_instruction(make_op("add"), softmax_parts["max"], log_sum_exp);

        // return a tuple of {O', LSE}
        target_mod.add_return({partial_output_o_prime, lse});
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& mm            = mpm.get_module();
        auto attn_group_ins = r.instructions["group"];
        auto* submod        = attn_group_ins->module_inputs().front();

        // TODO: for this pass of flash decoding, if LSE attn, do not do flash decoding
        auto return_ins = std::prev(submod->end());
        assert(return_ins->name() == "@return" and
               "Last instruction must be a @return instruction");
        if(return_ins->inputs().size() > 1)
            return;

        // get gemm1 and gemm2
        auto [gemm1, gemm2] = get_gemms(submod);

        // TODO: for this first pass of flash decoding, assuming no input fusion / not supporting
        auto q_param = gemm1->inputs()[0];
        auto k_param = gemm1->inputs()[1];
        auto v_param = gemm2->inputs()[1];
        assert(q_param->name() == "@param" and "Q should be a parameter");
        assert(k_param->name() == "@param" and "K should be a parameter");
        assert(v_param->name() == "@param" and "V should be a parameter");

        // check if N dimension is evenly divisible by num_splits
        if(k_param->get_shape().lens().back() % groups != 0)
            return;

        // get Q, V, K shapes from gemms
        auto qkv_shapes = get_qkv_shapes(q_param, k_param, v_param);

        // check shapes are ok and get flash decoding transformed shapes (Q', V', K')
        auto transform_info = get_transformed_shapes(qkv_shapes);

        // create mapping from submodule params to main module inputs
        auto group_inputs      = attn_group_ins->inputs();
        auto map_param_to_main = map_submod_params_to_inputs(submod, group_inputs);

        // get actual Q, K, V instructions from main module
        auto q = map_param_to_main.at(q_param);
        auto k = map_param_to_main.at(k_param);
        auto v = map_param_to_main.at(v_param);

        // insert reshape operations before group, for Q, K, V
        auto q_ndim    = q->get_shape().lens().size();
        int64_t g_axis = q_ndim - 2;

        // Q: [B, M, k] -> [B, G, M, k] via unsqueeze + broadcast
        auto q_unsqueeze =
            mm.insert_instruction(attn_group_ins, make_op("unsqueeze", {{"axes", {g_axis}}}), q);
        auto q_reshaped =
            mm.insert_instruction(attn_group_ins,
                                  make_op("multibroadcast", {{"out_lens", transform_info.q_shape}}),
                                  q_unsqueeze);

        // K: [B, k, N] -> [B, G, k, N/G] via reshape + transpose
        auto k_reshaped_intermediate = mm.insert_instruction(
            attn_group_ins, make_op("reshape", {{"dims", transform_info.k_intermediate}}), k);
        auto k_reshaped = mm.insert_instruction(
            attn_group_ins,
            make_op("transpose", {{"permutation", transform_info.k_transpose_perm}}),
            k_reshaped_intermediate);

        // V: [B, N, D] -> [B, G, N/G, D] via direct reshape
        auto v_reshaped = mm.insert_instruction(
            attn_group_ins, make_op("reshape", {{"dims", transform_info.v_shape}}), v);

        // create new input list by replacing Q, K, V with reshaped versions
        std::vector<instruction_ref> new_group_inputs = group_inputs;
        for(size_t i = 0; i < group_inputs.size(); ++i)
        {
            if(group_inputs[i] == q)
            {
                new_group_inputs[i] = q_reshaped;
            }
            else if(group_inputs[i] == k)
            {
                new_group_inputs[i] = k_reshaped;
            }
            else if(group_inputs[i] == v)
            {
                new_group_inputs[i] = v_reshaped;
            }
        }

        // create new submodule for flash decoding
        module m_flash_decode;
        m_flash_decode.set_bypass();

        // get parameter names
        auto q_name = q_param->get_operator().to_value()["parameter"].to<std::string>();
        auto k_name = k_param->get_operator().to_value()["parameter"].to<std::string>();
        auto v_name = v_param->get_operator().to_value()["parameter"].to<std::string>();

        // new params added first
        auto new_q_param = m_flash_decode.add_parameter(
            q_name, shape{qkv_shapes[0].type(), transform_info.q_shape});
        auto new_k_param = m_flash_decode.add_parameter(
            k_name, shape{qkv_shapes[1].type(), transform_info.k_shape});
        auto new_v_param = m_flash_decode.add_parameter(
            v_name, shape{qkv_shapes[2].type(), transform_info.v_shape});

        // build mapping for old params -> new params
        std::unordered_map<instruction_ref, instruction_ref> map_old_params_to_new;
        map_old_params_to_new[q_param] = new_q_param;
        map_old_params_to_new[k_param] = new_k_param;
        map_old_params_to_new[v_param] = new_v_param;

        // don't simply fuse previous attn submod, need to rebuild all the ops
        rebuild_attention_submodule(m_flash_decode, *submod, map_old_params_to_new);

        auto original_submod_name = attn_group_ins->module_inputs().front()->name();
        std::string new_mod_name  = original_submod_name + "_flash_decoding";

        module_ref mpm_flash_mod = mpm.create_module(new_mod_name, std::move(m_flash_decode));
        mpm_flash_mod->set_bypass();

        // insert the new group op, which returns a tuple of O' and LSE
        auto new_group_ins = mm.insert_instruction(attn_group_ins,
                                                   make_op("group", {{"tag", "attention"}}),
                                                   new_group_inputs,
                                                   {mpm_flash_mod});

        // unpack O' and LSE
        auto partial_output_o_prime = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 0}}), new_group_ins);
        auto lse = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 1}}), new_group_ins);

        // kernel 2
        // the partial outputs O'[g] are already weighted by their group's softmax,
        // LSE[g] contains log(sum(exp(S[g]))) for each group
        // To combine: weight by exp(LSE[g]) / sum_g(exp(LSE[g']))

        // compute global max for numerical stability
        auto lse_max =
            mm.insert_instruction(attn_group_ins, make_op("reduce_max", {{"axes", {g_axis}}}), lse);
        auto lse_max_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", lse->get_shape().lens()}}),
            lse_max);

        // exp(LSE - max_LSE)
        auto lse_sub = mm.insert_instruction(attn_group_ins, make_op("sub"), lse, lse_max_bcast);
        auto lse_exp = mm.insert_instruction(attn_group_ins, make_op("exp"), lse_sub);

        // sum across groups
        auto lse_sum = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {g_axis}}}), lse_exp);
        auto lse_sum_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", lse_exp->get_shape().lens()}}),
            lse_sum);

        // scale factor: exp(LSE[g] - max_LSE) / sum(exp(LSE - max_LSE))
        auto scale = mm.insert_instruction(attn_group_ins, make_op("div"), lse_exp, lse_sum_bcast);

        auto scale_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", partial_output_o_prime->get_shape().lens()}}),
            scale);

        // convert scale to match the type of partial_output_o_prime
        auto output_type     = partial_output_o_prime->get_shape().type();
        auto scale_converted = mm.insert_instruction(
            attn_group_ins, make_op("convert", {{"target_type", output_type}}), scale_bcast);

        // R = mul(O', broadcasted_scale)
        auto scaled_r = mm.insert_instruction(
            attn_group_ins, make_op("mul"), partial_output_o_prime, scale_converted);

        // O = sum(R, axis=G_axis)
        auto final_output_o = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {g_axis}}}), scaled_r);

        // squeeze G to match the original output shape
        auto final_squeezed_o = mm.insert_instruction(
            attn_group_ins, make_op("squeeze", {{"axes", {g_axis}}}), final_output_o);

        // replace the original group instruction with the final result
        mm.replace_instruction(attn_group_ins, final_squeezed_o);
    }
};

struct find_kv_cache_attention
{
    std::size_t* counter;

    auto matcher() const
    {
        static const std::unordered_set<std::string> skip_set = {
            "multibroadcast", "reshape", "unsqueeze"};

        auto keys =
            match::skip(match::name(skip_set))(match::name("concat_past_present")).bind("pres_k");
        auto k_transpose =
            match::skip(match::name(skip_set))(match::name("transpose")(match::arg(0)(keys)));
        auto queries = match::name("slice");
        auto gemm1   = match::name("dot")(match::arg(0)(queries), match::arg(1)(k_transpose));
        auto scale   = match::name("mul")(match::any_arg(0, 1)(gemm1));
        auto broadcasted_const = match::name("multibroadcast")(match::arg(0)(match::is_constant()));
        auto attn_scores       = match::any_of(scale, gemm1);
        auto causal_mask =
            match::name("where")(match::arg(0)(broadcasted_const), match::arg(2)(attn_scores));
        auto greater = match::name("greater")(match::arg(1)(match::any().bind("total_sl")));
        auto conv_greater =
            match::skip(match::name("unsqueeze"))(match::name("convert")(match::arg(0)(greater)));
        auto bc_greater         = match::name("multibroadcast")(match::arg(0)(conv_greater));
        auto mask               = match::name("where")(match::arg(0)(bc_greater),
                                         match::arg(2)(match::any_of(causal_mask, scale, gemm1)));
        auto attn_probabilities = match::skip(match::name("convert"))(
            match::softmax_input(match::skip(match::name("convert"))(mask)));
        auto values =
            match::skip(match::name(skip_set))(match::name("concat_past_present")).bind("pres_v");
        auto gemm2 = match::name("dot")(match::arg(0)(attn_probabilities), match::arg(1)(values));
        auto transpose_out = match::name("transpose")(match::arg(0)(gemm2));
        return match::name("reshape")(match::arg(0)(transpose_out));
    }

    std::string get_count() const { return std::to_string((*counter)++); }

    std::unordered_map<instruction_ref, instruction_ref>
    invert_map_ins(const std::unordered_map<instruction_ref, instruction_ref>& map_ins) const
    {
        std::unordered_map<instruction_ref, instruction_ref> inverse_map;
        for(auto const& [key, value] : map_ins)
        {
            assert(not contains(inverse_map, value));
            inverse_map[value] = key;
        }
        return inverse_map;
    }

    std::vector<instruction_ref>
    get_attn_instructions(module& m, instruction_ref start, instruction_ref end) const
    {
        std::queue<instruction_ref> inputs;
        std::unordered_set<instruction_ref> inss;
        inputs.push(end);

        static const std::unordered_set<std::string> valid_attn_ops = {"softmax",
                                                                       "broadcast",
                                                                       "dot",
                                                                       "slice",
                                                                       "transpose",
                                                                       "greater",
                                                                       "convert",
                                                                       "where",
                                                                       "reshape",
                                                                       "reduce_sum",
                                                                       "reduce_max",
                                                                       "broadcast",
                                                                       "multibroadcast",
                                                                       "@literal",
                                                                       "unsqueeze"};

        auto is_valid_attn_op = [&](auto i) {
            return i->get_operator().attributes().get("pointwise", false) or
                   contains(valid_attn_ops, i->get_operator().name()) or i == start or i == end;
        };

        while(not inputs.empty())
        {
            auto current_inp = inputs.front();
            inputs.pop();

            if(is_valid_attn_op(current_inp) and inss.insert(current_inp).second and
               current_inp != start)
            {
                for(auto i : current_inp->inputs())
                {
                    inputs.push(i);
                }
            }
        }
        std::vector<instruction_ref> sorted_inss(inss.begin(), inss.end());
        std::sort(
            sorted_inss.begin(), sorted_inss.end(), [&](instruction_ref x, instruction_ref y) {
                return std::distance(m.begin(), x) < std::distance(m.begin(), y);
            });
        return sorted_inss;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto total_sl = r.instructions["total_sl"];
        auto reshape  = r.result;

        // Capture all instructions part of the attention op
        auto attn_inss = get_attn_instructions(mpm.get_module(), total_sl, reshape);

        // Add captured instructions to new submodule
        module m_attn;
        std::unordered_map<instruction_ref, instruction_ref> map_mm_to_mattn;
        auto attn_outs = m_attn.fuse(attn_inss, &map_mm_to_mattn);

        for(auto ins : iterator_for(m_attn))
        {
            if(ins->can_eval())
            {
                auto lit_s   = ins->get_shape();
                auto strides = lit_s.strides();
                if(strides.size() == 4 and
                   std::all_of(
                       strides.begin(), strides.end() - 1, [](auto s) { return s == 0; }) and
                   strides.back() == 1)
                {
                    auto new_lit = m_attn.add_literal(
                        literal{shape{lit_s.type(), {lit_s.lens().back()}}, ins->eval().data()});
                    m_attn.replace_instruction(
                        ins, make_op("multibroadcast", {{"out_lens", lit_s.lens()}}), {new_lit});
                }
            }
        }
        dead_code_elimination{}.apply(m_attn);

        // Define outputs based on instructions that are used elsewhere in the graph
        std::vector<instruction_ref> required_outputs;
        std::copy_if(
            attn_inss.begin(), attn_inss.end(), std::back_inserter(required_outputs), [&](auto i) {
                return not std::all_of(i->outputs().begin(), i->outputs().end(), [&](auto o) {
                    return contains(attn_inss, o);
                });
            });

        assert(not required_outputs.empty());

        // Find corresponding output instructions in m_attn
        std::vector<instruction_ref> m_attn_outputs;
        std::transform(required_outputs.begin(),
                       required_outputs.end(),
                       std::back_inserter(m_attn_outputs),
                       [&](auto i) { return map_mm_to_mattn.at(i); });
        m_attn.add_return({m_attn_outputs.back()});

        // Define inputs to m_attn
        auto map_mattn_to_mm = invert_map_ins(map_mm_to_mattn);
        auto new_inputs      = m_attn.get_inputs(map_mattn_to_mm);

        module_ref mpm_attn = mpm.create_module("attn" + get_count(), std::move(m_attn));
        mpm_attn->set_bypass();

        // Construct group op with the attention module
        auto group_ins =
            mpm.get_module().insert_instruction(required_outputs.back(),
                                                make_op("group", {{"tag", "kv_cache_attention"}}),
                                                new_inputs,
                                                {mpm_attn});

        mpm.get_module().replace_instruction(required_outputs.back(), group_ins);
    }
};

/// Transforms KV cache to use paged attention with scatter/gather
/// This replaces concat_past_present with:
///   - scatter_none (for writing new K/V to block slots)
///   - gather (for reading K/V blocks into contiguous view)
/// Works directly on concat_past_present ops without requiring attention fusion
struct find_paged_attention
{
    paged_attention_config config;
    mutable std::unordered_set<instruction_ref> processed_concats;

    auto matcher() const
    {
        // Match concat_past_present directly in the main module
        return match::name("concat_past_present").bind("concat_kv");
    }

    /// Calculate the number of blocks needed for a given max sequence length
    std::size_t calculate_num_blocks(std::size_t max_seq_len) const
    {
        return (max_seq_len + config.tokens_per_block - 1) / config.tokens_per_block;
    }

    /// KV cache info extracted from a single concat_past_present
    struct kv_cache_info
    {
        instruction_ref concat_ins;        // The concat_past_present instruction
        instruction_ref past_kv_param;     // @param for past K or V cache
        instruction_ref new_kv;            // New K or V values input
        instruction_ref seqlens;           // Sequence lengths input
        std::size_t num_kv_heads;
        std::size_t max_seq_len;
        std::size_t head_dim;
        std::size_t batch_size;
    };

    std::optional<kv_cache_info> extract_kv_info(instruction_ref concat_ins) const
    {
        kv_cache_info info{};
        info.concat_ins = concat_ins;

        // concat_past_present takes (present, seqlens, past)
        auto inputs = concat_ins->inputs();
        if(inputs.size() != 3)
            return std::nullopt;

        info.new_kv       = inputs[0];  // present K or V
        info.seqlens      = inputs[1];  // sequence lengths
        info.past_kv_param = inputs[2]; // past K or V cache

        // Extract shape info from past cache: {batch, num_kv_heads, max_seq_len, head_dim}
        auto past_shape = info.past_kv_param->get_shape();
        auto past_lens  = past_shape.lens();

        if(past_lens.size() != 4)
            return std::nullopt;

        info.batch_size   = past_lens[0];
        info.num_kv_heads = past_lens[1];
        info.max_seq_len  = past_lens[2];
        info.head_dim     = past_lens[3];

        return info;
    }

    /// Reshape existing KV cache to paged format
    /// From: {batch, num_kv_heads, max_seq_len, head_dim}
    /// To:   {num_blocks, num_kv_heads, tokens_per_block, head_dim}
    instruction_ref reshape_to_paged(module& m,
                                     instruction_ref past_kv,
                                     instruction_ref insert_point,
                                     const kv_cache_info& info) const
    {
        std::size_t max_blocks_per_seq = calculate_num_blocks(info.max_seq_len);
        std::size_t total_blocks = info.batch_size * max_blocks_per_seq;

        auto reshaped = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(total_blocks),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.head_dim)}}}),
            past_kv);

        return reshaped;
    }

    /// Get or create control parameters (block_table, slot_mapping)
    /// These are shared across all concat_past_present ops
    struct control_params
    {
        instruction_ref block_table;
        instruction_ref slot_mapping;
    };

    control_params get_control_params(module& m, const kv_cache_info& info) const
    {
        control_params params{};

        std::size_t max_blocks_per_seq = calculate_num_blocks(info.max_seq_len);

        // Get num_new_tokens from new_kv shape: {batch, num_kv_heads, seq_len, head_dim}
        auto new_kv_shape = info.new_kv->get_shape();
        std::size_t num_new_tokens = new_kv_shape.lens()[0] * new_kv_shape.lens()[2];

        // Check if parameters already exist
        auto param_names = m.get_parameter_names();

        if(contains(param_names, "block_table"))
        {
            params.block_table = m.get_parameter("block_table");
        }
        else
        {
            shape block_table_shape{shape::int32_type, {info.batch_size, max_blocks_per_seq}};
            params.block_table = m.add_parameter("block_table", block_table_shape);
        }

        if(contains(param_names, "slot_mapping"))
        {
            params.slot_mapping = m.get_parameter("slot_mapping");
        }
        else
        {
            shape slot_mapping_shape{shape::int32_type, {num_new_tokens}};
            params.slot_mapping = m.add_parameter("slot_mapping", slot_mapping_shape);
        }

        return params;
    }

    /// Create scatter operation to write new K/V tokens to paged cache
    instruction_ref create_scatter_write(module& m,
                                         instruction_ref insert_point,
                                         instruction_ref cache_paged,
                                         instruction_ref slot_mapping,
                                         instruction_ref new_kv,
                                         const kv_cache_info& info) const
    {
        auto paged_shape = cache_paged->get_shape();
        auto paged_lens = paged_shape.lens();
        std::size_t num_blocks = paged_lens[0];
        std::size_t total_slots = num_blocks * config.tokens_per_block;

        // Flatten the paged cache for scatter: {total_slots, num_kv_heads, head_dim}
        auto cache_flat = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(total_slots),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            cache_paged);

        // Get new_kv shape: {batch, num_kv_heads, seq_len, head_dim}
        auto new_kv_shape = new_kv->get_shape();
        auto new_kv_lens = new_kv_shape.lens();
        std::size_t num_new_tokens = new_kv_lens[0] * new_kv_lens[2]; // batch * seq_len

        // Reshape new_kv for scatter: {num_new_tokens, num_kv_heads, head_dim}
        // First transpose to {batch, seq_len, num_kv_heads, head_dim}
        auto new_kv_transposed = m.insert_instruction(
            insert_point,
            make_op("transpose", {{"permutation", std::vector<int64_t>{0, 2, 1, 3}}}),
            new_kv);

        // Then reshape to {num_new_tokens, num_kv_heads, head_dim}
        auto new_kv_flat = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(num_new_tokens),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            new_kv_transposed);

        // Reshape slot_mapping to match updates rank: {num_new_tokens} -> {num_new_tokens, 1, 1}
        auto slot_mapping_reshaped = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(num_new_tokens),
                       static_cast<int64_t>(1),
                       static_cast<int64_t>(1)}}}),
            slot_mapping);

        // Broadcast to match updates shape: {num_new_tokens, num_kv_heads, head_dim}
        auto slot_mapping_broadcast = m.insert_instruction(
            insert_point,
            make_op("multibroadcast",
                    {{"out_lens",
                      {static_cast<int64_t>(num_new_tokens),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            slot_mapping_reshaped);

        // Scatter write: updates cache_flat at positions specified by slot_mapping
        auto scattered = m.insert_instruction(
            insert_point,
            make_op("scatter_none", {{"axis", 0}}),
            cache_flat,
            slot_mapping_broadcast,
            new_kv_flat);

        // Reshape back to paged shape: {num_blocks, num_kv_heads, tokens_per_block, head_dim}
        auto cache_updated = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(num_blocks),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.head_dim)}}}),
            scattered);

        return cache_updated;
    }

    /// Create gather operation to read K/V blocks for attention
    instruction_ref create_gather_read(module& m,
                                       instruction_ref insert_point,
                                       instruction_ref cache_paged,
                                       instruction_ref block_table,
                                       const kv_cache_info& info) const
    {
        auto block_table_shape = block_table->get_shape();
        auto block_table_lens = block_table_shape.lens();

        std::size_t batch_size = block_table_lens[0];
        std::size_t max_blocks_per_seq = block_table_lens[1];

        // Flatten block_table for gather: {batch * max_blocks_per_seq}
        auto block_table_flat = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims", {static_cast<int64_t>(batch_size * max_blocks_per_seq)}}}),
            block_table);

        // Gather blocks: cache_paged[block_table_flat]
        // Input: {num_blocks, num_kv_heads, tokens_per_block, head_dim}
        // Output: {batch * max_blocks_per_seq, num_kv_heads, tokens_per_block, head_dim}
        auto gathered = m.insert_instruction(
            insert_point,
            make_op("gather", {{"axis", 0}}),
            cache_paged,
            block_table_flat);

        // Reshape to {batch, max_blocks_per_seq, num_kv_heads, tokens_per_block, head_dim}
        auto gathered_reshaped = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(batch_size),
                       static_cast<int64_t>(max_blocks_per_seq),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.head_dim)}}}),
            gathered);

        // Transpose to {batch, num_kv_heads, max_blocks_per_seq, tokens_per_block, head_dim}
        auto gathered_transposed = m.insert_instruction(
            insert_point,
            make_op("transpose", {{"permutation", std::vector<int64_t>{0, 2, 1, 3, 4}}}),
            gathered_reshaped);

        // Reshape to contiguous view: {batch, num_kv_heads, max_seq_len, head_dim}
        std::size_t max_seq_from_blocks = max_blocks_per_seq * config.tokens_per_block;
        auto contiguous_view = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(batch_size),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(max_seq_from_blocks),
                       static_cast<int64_t>(info.head_dim)}}}),
            gathered_transposed);

        return contiguous_view;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m = mpm.get_module();
        auto concat_ins = r.instructions["concat_kv"];

        // Skip if already processed (we process each concat independently)
        if(processed_concats.count(concat_ins) > 0)
            return;

        // Extract info from this concat_past_present
        auto info_opt = extract_kv_info(concat_ins);
        if(!info_opt)
            return;

        auto info = *info_opt;

        // Mark as processed
        processed_concats.insert(concat_ins);

        // Get or create control parameters (shared across all concat ops)
        auto ctrl_params = get_control_params(m, info);

        // Reshape past cache to paged format (insert before concat_ins)
        auto cache_paged = reshape_to_paged(m, info.past_kv_param, concat_ins, info);

        // Create scatter to write new tokens (insert before concat_ins)
        auto cache_updated = create_scatter_write(
            m, concat_ins, cache_paged, ctrl_params.slot_mapping, info.new_kv, info);

        // Create gather to read contiguous view (insert before concat_ins)
        auto gathered = create_gather_read(
            m, concat_ins, cache_updated, ctrl_params.block_table, info);

        // Replace concat_past_present with gathered result
        m.replace_instruction(concat_ins, gathered);
    }
};

/// Paged attention with combined KV cache format (dimension 2 for K/V separation)
/// This format is compatible with external KV cache manager APIs.
/// 
/// Cache shapes:
///   - Combined KV cache: {2, num_blocks, tokens_per_block, num_kv_heads, head_dim}
///                         ^-- 0 = Key blocks, 1 = Value blocks
///   - Block table: {batch_size, 2, max_blocks_per_seq}
///                               ^-- 0 = K block indices, 1 = V block indices
///
/// This pass matches pairs of concat_past_present (K and V) and transforms them
/// to use a single combined paged cache with scatter/gather operations.
struct find_paged_attention_combined
{
    paged_attention_config config;
    mutable std::unordered_set<instruction_ref> processed_concats;

    auto matcher() const
    {
        // Match concat_past_present directly - we'll pair K and V in apply
        return match::name("concat_past_present").bind("concat_kv");
    }

    std::size_t calculate_num_blocks(std::size_t max_seq_len) const
    {
        return (max_seq_len + config.tokens_per_block - 1) / config.tokens_per_block;
    }

    /// KV cache info for a single concat_past_present
    struct kv_cache_info
    {
        instruction_ref concat_ins;
        instruction_ref past_kv_param;
        instruction_ref new_kv;
        instruction_ref seqlens;
        std::size_t num_kv_heads;
        std::size_t max_seq_len;
        std::size_t head_dim;
        std::size_t batch_size;
    };

    /// Combined KV pair info
    struct kv_pair_info
    {
        kv_cache_info k_info;
        kv_cache_info v_info;
    };

    std::optional<kv_cache_info> extract_kv_info(instruction_ref concat_ins) const
    {
        kv_cache_info info{};
        info.concat_ins = concat_ins;

        auto inputs = concat_ins->inputs();
        if(inputs.size() != 3)
            return std::nullopt;

        info.new_kv        = inputs[0];
        info.seqlens       = inputs[1];
        info.past_kv_param = inputs[2];

        auto past_shape = info.past_kv_param->get_shape();
        auto past_lens  = past_shape.lens();

        if(past_lens.size() != 4)
            return std::nullopt;

        info.batch_size   = past_lens[0];
        info.num_kv_heads = past_lens[1];
        info.max_seq_len  = past_lens[2];
        info.head_dim     = past_lens[3];

        return info;
    }

    /// Find the paired concat_past_present for V given K's concat
    /// Returns nullopt if no valid pair found
    std::optional<instruction_ref> find_paired_concat(module& m, instruction_ref k_concat) const
    {
        // Look for another concat_past_present that shares the same seqlens input
        auto k_seqlens = k_concat->inputs()[1];
        
        for(auto ins : iterator_for(m))
        {
            if(ins->name() != "concat_past_present")
                continue;
            if(ins == k_concat)
                continue;
            if(processed_concats.count(ins) > 0)
                continue;
            
            // Check if they share the same seqlens
            if(ins->inputs().size() == 3 and ins->inputs()[1] == k_seqlens)
            {
                return ins;
            }
        }
        return std::nullopt;
    }

    /// Control parameters for combined KV cache
    /// Block table shape: {batch_size, 2, max_blocks_per_seq}
    ///                                 ^-- 0 = K indices, 1 = V indices
    struct control_params_combined
    {
        instruction_ref block_table;   // {batch, 2, max_blocks}
        instruction_ref slot_mapping;  // {num_new_tokens}
    };

    control_params_combined get_control_params(module& m, const kv_cache_info& info) const
    {
        control_params_combined params{};
        std::size_t max_blocks_per_seq = calculate_num_blocks(info.max_seq_len);

        auto new_kv_shape     = info.new_kv->get_shape();
        std::size_t num_new_tokens = new_kv_shape.lens()[0] * new_kv_shape.lens()[2];

        auto param_names = m.get_parameter_names();

        if(contains(param_names, "block_table"))
        {
            params.block_table = m.get_parameter("block_table");
        }
        else
        {
            // Shape: {batch_size, 2, max_blocks_per_seq}
            // The dimension of 2 separates K and V block indices
            shape block_table_shape{shape::int32_type,
                                    {info.batch_size, 2, max_blocks_per_seq}};
            params.block_table = m.add_parameter("block_table", block_table_shape);
        }

        if(contains(param_names, "slot_mapping"))
        {
            params.slot_mapping = m.get_parameter("slot_mapping");
        }
        else
        {
            shape slot_mapping_shape{shape::int32_type, {num_new_tokens}};
            params.slot_mapping = m.add_parameter("slot_mapping", slot_mapping_shape);
        }

        return params;
    }

    /// Reshape separate K and V past caches to combined paged format
    /// Input K: {batch, num_kv_heads, max_seq_len, head_dim}
    /// Input V: {batch, num_kv_heads, max_seq_len, head_dim}
    /// Output: {2, num_blocks, tokens_per_block, num_kv_heads, head_dim}
    ///
    /// The transformation requires both reshape AND transpose because:
    /// - Input layout: [batch, heads, seq, dim] - all seq positions per head are contiguous
    /// - Output layout: [batch, blocks, tokens, heads, dim] - enables contiguous seq when flattened
    /// 
    /// Steps:
    /// 1. Reshape: {batch, heads, seq, dim} -> {batch, heads, blocks, tokens_per_block, dim}
    /// 2. Transpose: {batch, heads, blocks, tokens, dim} -> {batch, blocks, tokens, heads, dim}
    instruction_ref reshape_to_combined_paged(module& m,
                                              instruction_ref past_k,
                                              instruction_ref past_v,
                                              instruction_ref insert_point,
                                              const kv_cache_info& info) const
    {
        std::size_t max_blocks_per_seq = calculate_num_blocks(info.max_seq_len);
        std::size_t total_blocks       = info.batch_size * max_blocks_per_seq;

        // Step 1: Reshape K from {batch, heads, seq, dim} to {batch, heads, blocks, tokens_per_block, dim}
        // This splits the sequence dimension into blocks and tokens_per_block
        auto k_reshaped = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(info.batch_size),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(max_blocks_per_seq),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.head_dim)}}}),
            past_k);

        // Step 2: Transpose K from {batch, heads, blocks, tokens, dim} to {batch, blocks, tokens, heads, dim}
        // Permutation {0, 2, 3, 1, 4} moves heads after tokens so that flattening blocks*tokens gives seq
        auto k_transposed = m.insert_instruction(
            insert_point,
            make_op("transpose", {{"permutation", std::vector<int64_t>{0, 2, 3, 1, 4}}}),
            k_reshaped);

        // Reshape to flatten batch into blocks: {1, total_blocks, tokens_per_block, heads, dim}
        auto k_paged = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(1),
                       static_cast<int64_t>(total_blocks),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            k_transposed);

        // Repeat for V
        auto v_reshaped = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(info.batch_size),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(max_blocks_per_seq),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.head_dim)}}}),
            past_v);

        auto v_transposed = m.insert_instruction(
            insert_point,
            make_op("transpose", {{"permutation", std::vector<int64_t>{0, 2, 3, 1, 4}}}),
            v_reshaped);

        auto v_paged = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(1),
                       static_cast<int64_t>(total_blocks),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            v_transposed);

        // Concatenate along dimension 0 to get {2, blocks, tokens, heads, dim}
        auto kv_combined = m.insert_instruction(
            insert_point, make_op("concat", {{"axis", 0}}), k_paged, v_paged);

        return kv_combined;
    }

    /// Create scatter write for combined KV cache
    /// Writes new K and V tokens to their respective slots in the combined cache
    instruction_ref create_scatter_write_combined(module& m,
                                                  instruction_ref insert_point,
                                                  instruction_ref cache_combined,
                                                  instruction_ref slot_mapping,
                                                  instruction_ref new_k,
                                                  instruction_ref new_v,
                                                  const kv_cache_info& info) const
    {
        auto paged_shape       = cache_combined->get_shape();
        auto paged_lens        = paged_shape.lens();
        std::size_t num_blocks = paged_lens[1];
        std::size_t total_slots = num_blocks * config.tokens_per_block;

        // Flatten combined cache for scatter: {2, total_slots, num_kv_heads, head_dim}
        auto cache_flat = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(2),
                       static_cast<int64_t>(total_slots),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            cache_combined);

        // Slice to get K cache: {1, total_slots, num_kv_heads, head_dim}
        auto k_cache_slice = m.insert_instruction(
            insert_point,
            make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
            cache_flat);

        // Slice to get V cache: {1, total_slots, num_kv_heads, head_dim}
        auto v_cache_slice = m.insert_instruction(
            insert_point,
            make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
            cache_flat);

        // Squeeze the leading dimension for scatter
        auto k_cache_squeezed = m.insert_instruction(
            insert_point, make_op("squeeze", {{"axes", {0}}}), k_cache_slice);

        auto v_cache_squeezed = m.insert_instruction(
            insert_point, make_op("squeeze", {{"axes", {0}}}), v_cache_slice);

        // Prepare new K for scatter
        auto new_k_shape              = new_k->get_shape();
        auto new_k_lens               = new_k_shape.lens();
        std::size_t num_new_tokens = new_k_lens[0] * new_k_lens[2];

        // Transpose new K: {batch, heads, seq, dim} -> {batch, seq, heads, dim}
        auto new_k_transposed = m.insert_instruction(
            insert_point,
            make_op("transpose", {{"permutation", std::vector<int64_t>{0, 2, 1, 3}}}),
            new_k);

        // Reshape to {num_new_tokens, num_kv_heads, head_dim}
        auto new_k_flat = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(num_new_tokens),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            new_k_transposed);

        // Same for new V
        auto new_v_transposed = m.insert_instruction(
            insert_point,
            make_op("transpose", {{"permutation", std::vector<int64_t>{0, 2, 1, 3}}}),
            new_v);

        auto new_v_flat = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(num_new_tokens),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            new_v_transposed);

        // Prepare slot_mapping for broadcast
        auto slot_mapping_reshaped = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(num_new_tokens),
                       static_cast<int64_t>(1),
                       static_cast<int64_t>(1)}}}),
            slot_mapping);

        auto slot_mapping_broadcast = m.insert_instruction(
            insert_point,
            make_op("multibroadcast",
                    {{"out_lens",
                      {static_cast<int64_t>(num_new_tokens),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            slot_mapping_reshaped);

        // Scatter K
        auto k_scattered = m.insert_instruction(insert_point,
                                                make_op("scatter_none", {{"axis", 0}}),
                                                k_cache_squeezed,
                                                slot_mapping_broadcast,
                                                new_k_flat);

        // Scatter V
        auto v_scattered = m.insert_instruction(insert_point,
                                                make_op("scatter_none", {{"axis", 0}}),
                                                v_cache_squeezed,
                                                slot_mapping_broadcast,
                                                new_v_flat);

        // Unsqueeze back to add leading dimension
        auto k_updated = m.insert_instruction(
            insert_point, make_op("unsqueeze", {{"axes", {0}}}), k_scattered);

        auto v_updated = m.insert_instruction(
            insert_point, make_op("unsqueeze", {{"axes", {0}}}), v_scattered);

        // Concat back to combined format: {2, total_slots, heads, dim}
        auto cache_updated_flat =
            m.insert_instruction(insert_point, make_op("concat", {{"axis", 0}}), k_updated, v_updated);

        // Reshape back to paged shape: {2, num_blocks, tokens_per_block, heads, dim}
        auto cache_updated = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(2),
                       static_cast<int64_t>(num_blocks),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            cache_updated_flat);

        return cache_updated;
    }

    /// Create gather read for combined KV cache
    /// block_table shape: {batch, 2, max_blocks_per_seq}
    /// Returns separate K and V views for attention computation
    struct gathered_kv_result
    {
        instruction_ref k_view;  // {batch, num_kv_heads, max_seq_len, head_dim}
        instruction_ref v_view;  // {batch, num_kv_heads, max_seq_len, head_dim}
    };

    gathered_kv_result create_gather_read_combined(module& m,
                                                   instruction_ref insert_point,
                                                   instruction_ref cache_combined,
                                                   instruction_ref block_table,
                                                   const kv_cache_info& info) const
    {
        auto block_table_shape           = block_table->get_shape();
        auto block_table_lens            = block_table_shape.lens();
        std::size_t batch_size           = block_table_lens[0];
        std::size_t max_blocks_per_seq = block_table_lens[2];

        // Slice block_table to get K indices: {batch, 1, max_blocks} -> {batch, max_blocks}
        auto k_block_table_slice = m.insert_instruction(
            insert_point,
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}),
            block_table);

        auto k_block_table = m.insert_instruction(
            insert_point, make_op("squeeze", {{"axes", {1}}}), k_block_table_slice);

        // Slice block_table to get V indices
        auto v_block_table_slice = m.insert_instruction(
            insert_point,
            make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}),
            block_table);

        auto v_block_table = m.insert_instruction(
            insert_point, make_op("squeeze", {{"axes", {1}}}), v_block_table_slice);

        // Cache layout from scatter: {2, blocks, tokens, heads, dim}
        // Slice combined cache to get K cache: {1, blocks, tokens, heads, dim} -> {blocks, tokens, heads, dim}
        auto k_cache_slice = m.insert_instruction(
            insert_point,
            make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}),
            cache_combined);

        auto k_cache = m.insert_instruction(
            insert_point, make_op("squeeze", {{"axes", {0}}}), k_cache_slice);

        // Slice to get V cache
        auto v_cache_slice = m.insert_instruction(
            insert_point,
            make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}),
            cache_combined);

        auto v_cache = m.insert_instruction(
            insert_point, make_op("squeeze", {{"axes", {0}}}), v_cache_slice);

        // Flatten block tables for gather: {batch * max_blocks}
        auto k_block_flat = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims", {static_cast<int64_t>(batch_size * max_blocks_per_seq)}}}),
            k_block_table);

        auto v_block_flat = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims", {static_cast<int64_t>(batch_size * max_blocks_per_seq)}}}),
            v_block_table);

        // Gather K blocks along axis 0 (blocks dimension)
        // k_cache: {blocks, tokens, heads, dim}
        // Result: {batch * max_blocks, tokens, heads, dim}
        auto k_gathered =
            m.insert_instruction(insert_point, make_op("gather", {{"axis", 0}}), k_cache, k_block_flat);

        // Gather V blocks
        auto v_gathered =
            m.insert_instruction(insert_point, make_op("gather", {{"axis", 0}}), v_cache, v_block_flat);

        // Reshape K from {batch * max_blocks, tokens, heads, dim} to {batch, max_blocks, tokens, heads, dim}
        auto k_reshaped = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(batch_size),
                       static_cast<int64_t>(max_blocks_per_seq),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            k_gathered);

        // Transpose to {batch, heads, max_blocks, tokens, dim}
        // Permutation {0, 3, 1, 2, 4} moves heads from position 3 to position 1
        auto k_transposed = m.insert_instruction(
            insert_point,
            make_op("transpose", {{"permutation", std::vector<int64_t>{0, 3, 1, 2, 4}}}),
            k_reshaped);

        // Reshape to {batch, heads, seq, dim} by merging blocks and tokens
        std::size_t max_seq_from_blocks = max_blocks_per_seq * config.tokens_per_block;
        auto k_merged                   = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(batch_size),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(max_seq_from_blocks),
                       static_cast<int64_t>(info.head_dim)}}}),
            k_transposed);

        // Make contiguous - required for attention kernel
        auto k_view =
            m.insert_instruction(insert_point, make_op("contiguous"), k_merged);

        // Same for V
        // Reshape from {batch * max_blocks, tokens, heads, dim} to {batch, max_blocks, tokens, heads, dim}
        auto v_reshaped = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(batch_size),
                       static_cast<int64_t>(max_blocks_per_seq),
                       static_cast<int64_t>(config.tokens_per_block),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(info.head_dim)}}}),
            v_gathered);

        // Transpose to {batch, heads, max_blocks, tokens, dim}
        auto v_transposed = m.insert_instruction(
            insert_point,
            make_op("transpose", {{"permutation", std::vector<int64_t>{0, 3, 1, 2, 4}}}),
            v_reshaped);

        // Reshape to {batch, heads, seq, dim}
        auto v_merged = m.insert_instruction(
            insert_point,
            make_op("reshape",
                    {{"dims",
                      {static_cast<int64_t>(batch_size),
                       static_cast<int64_t>(info.num_kv_heads),
                       static_cast<int64_t>(max_seq_from_blocks),
                       static_cast<int64_t>(info.head_dim)}}}),
            v_transposed);

        // Make contiguous
        auto v_view =
            m.insert_instruction(insert_point, make_op("contiguous"), v_merged);

        return {k_view, v_view};
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& m         = mpm.get_module();
        auto concat_ins = r.instructions["concat_kv"];

        // Skip if already processed
        if(processed_concats.count(concat_ins) > 0)
            return;

        // Extract info from this concat (assume it's K)
        auto k_info_opt = extract_kv_info(concat_ins);
        if(not k_info_opt)
            return;

        auto k_info = *k_info_opt;

        // Find paired V concat
        auto v_concat_opt = find_paired_concat(m, concat_ins);
        if(not v_concat_opt)
        {
            // No pair found, fall back to separate processing
            // (or skip - depending on desired behavior)
            return;
        }

        auto v_concat   = *v_concat_opt;
        auto v_info_opt = extract_kv_info(v_concat);
        if(not v_info_opt)
            return;

        auto v_info = *v_info_opt;

        // Mark both as processed
        processed_concats.insert(concat_ins);
        processed_concats.insert(v_concat);

        // Get control parameters with combined block table format
        auto ctrl_params = get_control_params(m, k_info);

        // Determine insert point (use the later of the two concats)
        auto k_pos = std::distance(m.begin(), concat_ins);
        auto v_pos = std::distance(m.begin(), v_concat);
        auto insert_point = (k_pos > v_pos) ? concat_ins : v_concat;

        // Reshape K and V to combined paged format
        auto cache_combined = reshape_to_combined_paged(
            m, k_info.past_kv_param, v_info.past_kv_param, insert_point, k_info);

        // Scatter write new K and V tokens
        auto cache_updated = create_scatter_write_combined(m,
                                                           insert_point,
                                                           cache_combined,
                                                           ctrl_params.slot_mapping,
                                                           k_info.new_kv,
                                                           v_info.new_kv,
                                                           k_info);

        // Gather read for attention
        auto [k_view, v_view] =
            create_gather_read_combined(m, insert_point, cache_updated, ctrl_params.block_table, k_info);

        // Replace the original concat_past_present instructions
        m.replace_instruction(concat_ins, k_view);
        m.replace_instruction(v_concat, v_view);
    }
};

} // namespace

void fuse_attention::apply(module_pass_manager& mpm) const
{
    std::size_t counter = 0;

    // Fuse kv-cache attention by default
    match::find_matches(mpm, find_kv_cache_attention{.counter = &counter});
    mpm.get_module().sort();
    mpm.run_pass(dead_code_elimination{});
    

    // Transform to paged attention if enabled
    // Matches concat_past_present directly, so can run before or after find_kv_cache_attention
    if(enabled(MIGRAPHX_ENABLE_PAGED_ATTN{}))
    {
        if(paged_attn_config.use_combined_kv)
        {
            // Use combined KV format with dimension 2 for K/V separation
            // Compatible with external KV cache manager APIs
            match::find_matches(mpm, find_paged_attention_combined{.config = paged_attn_config, 
                                                                   .processed_concats = {}});
        }
        else
        {
            // Use separate K and V tensors (default MIGraphX behavior)
            match::find_matches(mpm, find_paged_attention{.config = paged_attn_config, 
                                                          .processed_concats = {}});
        }
        mpm.get_module().sort();
        mpm.run_pass(dead_code_elimination{});
    }

    // Only fuse plain attention when requested
    if(attn_enabled)
    {
        match::find_matches(mpm, find_attention{.counter = &counter});
        mpm.get_module().sort();
        mpm.run_pass(dead_code_elimination{});
    }

    std::size_t num_splits = 0;
    if(flash_decoding_num_splits.has_value())
    {
        // Use the value from the constructor (for testing)
        num_splits = *flash_decoding_num_splits;
    }
    else
    {
        // Default behavior: read from the env var (for non-test use)
        num_splits = get_num_splits();
    }
    if(num_splits > 1)
    {
        match::find_matches(mpm, find_flash_decoding{.groups = num_splits});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
