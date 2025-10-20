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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_FLASH_DECODING);

static std::size_t get_num_splits()
{
    std::string value = string_value_of(MIGRAPHX_ENABLE_FLASH_DECODING{}, "0");
    try
    {
        return std::stoul(value);
    }
    catch(const std::exception&)
    {
        return 0;
    }
}

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
    std::size_t G;

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
        assert(gemms.size() == 2 && "Expected exactly 2 gemm operations in attention submodule");
        
        // gemms[0] is Q@K, gemms[1] is P@V
        // gemms are in order since we iterate from begin to end
        return {gemms[0], gemms[1]};
    }

    std::vector<shape> get_QKV_shapes(instruction_ref Q, instruction_ref K, instruction_ref V) const 
    {
        std::vector<shape> QKV_shapes;
        auto Q_shape = Q->get_shape();
        auto K_shape = K->get_shape();
        auto V_shape = V->get_shape();

        QKV_shapes.push_back(Q_shape);
        QKV_shapes.push_back(K_shape);
        QKV_shapes.push_back(V_shape);

        assert((Q_shape.lens().size() == 3 || Q_shape.lens().size() == 4) && "Expected 3D or 4D Q, K, V shapes");
        assert(K_shape.lens().size() == Q_shape.lens().size() && 
               V_shape.lens().size() == Q_shape.lens().size() && 
               "Q, K, V must have same number of dimensions");
        return QKV_shapes;
    }

    std::vector<std::vector<size_t>> 
    get_transformed_shapes(const std::vector<shape>& input_shapes) const
    {
        assert(input_shapes.size() == 3 && "Expected Q, K, V shapes");

        auto Q_lens = input_shapes[0].lens();
        auto K_lens = input_shapes[1].lens();
        auto V_lens = input_shapes[2].lens();

        // 3D: Q_lens = [B, M, k]
        // 4D: Q_lens = [B, H, M, k]
        size_t ndim = Q_lens.size();
        size_t N    = K_lens[ndim - 1];
        
        // TODO: do we wanna support this differently? 
        assert(N % G == 0 && "N must be divisible by G");
        size_t N_split = N / G;

        // batch_dims + G + spatial_dims
        auto insert_G = [&](const auto& lens) {
            std::vector<size_t> new_lens(lens.begin(), lens.begin() + ndim - 2);  // batch dims
            new_lens.push_back(G);                                                // insert G
            new_lens.insert(new_lens.end(), lens.begin() + ndim - 2, lens.end()); // last 2 dims
            return new_lens;
        };

        auto Q_new = insert_G(Q_lens);
        auto K_new = insert_G(K_lens);
        auto V_new = insert_G(V_lens);

        // update N -> N/G in K and V
        K_new[K_new.size() - 1] = N_split;  // K: [..., G, k, N/G]
        V_new[V_new.size() - 2] = N_split;  // V: [..., G, N/G, D]

        return {Q_new, K_new, V_new};
    }

    std::unordered_map<instruction_ref, instruction_ref> 
    map_submod_params_to_inputs(module_ref submod, 
                                const std::vector<instruction_ref>& group_inputs) const 
    {
        std::unordered_map<instruction_ref, instruction_ref> map_param_to_main;

        auto param_names = submod->get_parameter_names();
        assert(param_names.size() == group_inputs.size() && 
               "Number of parameters must match number of inputs");

        for(size_t i = 0; i < param_names.size(); ++i) 
        {
            auto param_ins               = submod->get_parameter(param_names[i]);
            map_param_to_main[param_ins] = group_inputs[i];
        }

        // verify the mapping is correct
        auto expected_inputs = submod->get_inputs(map_param_to_main);
        assert(expected_inputs == group_inputs && "Mapped inputs don't match group inputs");
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
                               assert(contains(map_old_to_new, i) && "Input not found in map");
                               return map_old_to_new.at(i);
                           });

            auto op = ins->get_operator();

            // transform operators that depend on tensor shape/rank
            // adjust reduction axes for the new rank
            if(op.name() == "reduce_max" or op.name() == "reduce_sum")
            {
                auto original_axes = op.to_value()["axes"].to_vector<int64_t>();
                assert(original_axes.size() == 1 && "Expected single axis for reduction");

                const auto& new_input_shape      = new_inputs.front()->get_shape();
                assert(original_axes.front() == 
                           static_cast<int64_t>(ins->inputs().front()->get_shape().lens().size() - 1) or 
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
                assert(sibling != parent->inputs().end() && 
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
        auto partial_output_O_prime = map_old_to_new.at(orig_return_ins);

        // calculate LSE = max(S) + log(sum(exp(S - max(S))))
        assert(contains(softmax_parts, "max") and contains(softmax_parts, "sum_exp"));
        auto log_sum_exp = target_mod.add_instruction(make_op("log"), softmax_parts["sum_exp"]);
        auto lse = target_mod.add_instruction(make_op("add"), softmax_parts["max"], log_sum_exp);

        // return a tuple of {O', LSE}
        target_mod.add_return({partial_output_O_prime, lse});
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto& mm = mpm.get_module();
        auto attn_group_ins = r.instructions["group"];
        auto submod = attn_group_ins->module_inputs().front();

        // TODO: for this pass of flash decoding, if LSE attn, do not do flash decoding
        auto return_ins = std::prev(submod->end());
        if(return_ins->inputs().size() > 1)
            return;

        // get gemm1 and gemm2
        auto [gemm1, gemm2] = get_gemms(submod);

        // TODO: for this first pass of flash decoding, assuming no input fusion / not suppporting
        auto Q_param = gemm1->inputs()[0];
        auto K_param = gemm1->inputs()[1];
        auto V_param = gemm2->inputs()[1];
        assert(Q_param->name() == "@param" && "Q should be a parameter");
        assert(K_param->name() == "@param" && "K should be a parameter");
        assert(V_param->name() == "@param" && "V should be a parameter");

        // get Q, V, K shapes from gemms
        auto QKV_shapes = get_QKV_shapes(Q_param, K_param, V_param);

        // check shapes are ok and get flash decoding transformed shapes (Q', V', K')
        auto transformed_shapes = get_transformed_shapes(QKV_shapes);
        auto& Qp_shape          = transformed_shapes[0];
        auto& Kp_shape          = transformed_shapes[1];
        auto& Vp_shape          = transformed_shapes[2];

        // create mapping from submodule params to main module inputs
        auto group_inputs      = attn_group_ins->inputs();
        auto map_param_to_main = map_submod_params_to_inputs(submod, group_inputs);

        // get actual Q, K, V instructions from main module
        auto Q = map_param_to_main.at(Q_param);
        auto K = map_param_to_main.at(K_param);
        auto V = map_param_to_main.at(V_param);

        // insert reshape operations before group, for Q, K, V
        auto q_ndim    = Q->get_shape().lens().size();
        int64_t G_axis = q_ndim - 2;

        auto Q_unsqueeze = 
            mm.insert_instruction(attn_group_ins, make_op("unsqueeze", {{"axes", {G_axis}}}), Q);

        auto Q_reshaped = 
            mm.insert_instruction(attn_group_ins, make_op("multibroadcast", {{"out_lens", Qp_shape}}), Q_unsqueeze);
                
        auto K_reshaped = 
            mm.insert_instruction(attn_group_ins, make_op("reshape", {{"dims", Kp_shape}}), K);
                
        auto V_reshaped = 
            mm.insert_instruction(attn_group_ins, make_op("reshape", {{"dims", Vp_shape}}), V);

        // create new input list by replacing Q, K, V with reshaped versions
        std::vector<instruction_ref> new_group_inputs = group_inputs;
        for(size_t i = 0; i < group_inputs.size(); ++i) 
        {
            if(group_inputs[i] == Q) 
            {
                new_group_inputs[i] = Q_reshaped;
            } else if(group_inputs[i] == K) 
            {
                new_group_inputs[i] = K_reshaped;
            } else if(group_inputs[i] == V) 
            {
                new_group_inputs[i] = V_reshaped;
            }
        }
        
        // create new submodule for flash decoding
        module m_flash_decode;
        m_flash_decode.set_bypass();

        // get parameter names
        auto Q_name = Q_param->get_operator().to_value()["parameter"].to<std::string>();
        auto K_name = K_param->get_operator().to_value()["parameter"].to<std::string>();
        auto V_name = V_param->get_operator().to_value()["parameter"].to<std::string>();

        // new params added first
        auto new_Q_param = 
            m_flash_decode.add_parameter(Q_name, shape{QKV_shapes[0].type(), Qp_shape});
        auto new_K_param = 
            m_flash_decode.add_parameter(K_name, shape{QKV_shapes[1].type(), Kp_shape});
        auto new_V_param = 
            m_flash_decode.add_parameter(V_name, shape{QKV_shapes[2].type(), Vp_shape});
        
        // build mapping for old params -> new params
        std::unordered_map<instruction_ref, instruction_ref> map_old_params_to_new;
        map_old_params_to_new[Q_param] = new_Q_param;
        map_old_params_to_new[K_param] = new_K_param;
        map_old_params_to_new[V_param] = new_V_param;

        // don't simply fuse previous attn submod, need to rebuild all the ops
        rebuild_attention_submodule(m_flash_decode, *submod, map_old_params_to_new);

        auto original_submod_name = attn_group_ins->module_inputs().front()->name();
        std::string new_mod_name = original_submod_name + "_flash_decoding";

        module_ref mpm_flash_mod = mpm.create_module(new_mod_name, std::move(m_flash_decode));
        mpm_flash_mod->set_bypass();

        // insert the new group op, which returns a tuple of O' and LSE
        auto new_group_ins = mm.insert_instruction(attn_group_ins,
                                                   make_op("group", {{"tag", "attention"}}),
                                                   new_group_inputs,
                                                   {mpm_flash_mod});

        // unpack O' and LSE
        auto partial_output_O_prime = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 0}}), new_group_ins);
        auto lse = mm.insert_instruction(
            attn_group_ins, make_op("get_tuple_elem", {{"index", 1}}), new_group_ins);

        // kernel 2
        // scale = softmax(L, axis=G_axis)
        auto lse_max = 
            mm.insert_instruction(attn_group_ins, make_op("reduce_max", {{"axes", {G_axis}}}), lse);
        auto lse_max_bcast = mm.insert_instruction(
            attn_group_ins, 
            make_op("multibroadcast", {{"out_lens", lse->get_shape().lens()}}), 
            lse_max);
        auto lse_sub = mm.insert_instruction(attn_group_ins, make_op("sub"), lse, lse_max_bcast);
        auto lse_exp = mm.insert_instruction(attn_group_ins, make_op("exp"), lse_sub);
        auto lse_sum = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {G_axis}}}), lse_exp);
        auto lse_sum_bcast = mm.insert_instruction(
            attn_group_ins, 
            make_op("multibroadcast", {{"out_lens", lse_exp->get_shape().lens()}}), 
            lse_sum);
        auto scale = mm.insert_instruction(attn_group_ins, make_op("div"), lse_exp, lse_sum_bcast);

        auto scale_bcast = mm.insert_instruction(
            attn_group_ins,
            make_op("multibroadcast", {{"out_lens", partial_output_O_prime->get_shape().lens()}}),
            scale);

        // R = mul(O', broadcasted_scale)
        auto scaled_R = mm.insert_instruction(
            attn_group_ins, make_op("mul"), partial_output_O_prime, scale_bcast);

        // O = sum(R, axis=G_axis)
        auto final_output_O = mm.insert_instruction(
            attn_group_ins, make_op("reduce_sum", {{"axes", {G_axis}}}), scaled_R);

        // squeeze G to match the original output shape
        auto final_squeezed_O = mm.insert_instruction(
            attn_group_ins, make_op("squeeze", {{"axes", {G_axis}}}), final_output_O);

        // replace the original group instruction with the final result
        mm.replace_instruction(attn_group_ins, final_squeezed_O);
    }
};

} // namespace

void fuse_attention::apply(module_pass_manager& mpm) const
{
    std::size_t counter = 0;
    match::find_matches(mpm, find_attention{.counter = &counter});
    mpm.get_module().sort();
    mpm.run_pass(dead_code_elimination{});

    std::size_t num_splits = get_num_splits();
    if(num_splits > 1) {
        match::find_matches(mpm, find_flash_decoding{.G = num_splits});
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
