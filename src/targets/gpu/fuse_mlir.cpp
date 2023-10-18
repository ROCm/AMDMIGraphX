/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_EXTRA_MLIR);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MLIR);

bool mlir_enabled()
{
#ifdef MIGRAPHX_MLIR
    const bool mlir_disabled = enabled(MIGRAPHX_DISABLE_MLIR{});
    return not mlir_disabled;
#else
    return false;
#endif
}

#ifdef MIGRAPHX_MLIR

struct mlir_op
{
    std::string name() const { return "gpu::mlir_op"; }
    operation op = make_op("convolution");

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.op, "op"));
    }

    shape compute_shape(std::vector<shape> inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.packed_or_broadcasted();
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");

        module_ref mod = mods[0];
        auto type      = mod->get_output_shapes().front().type();
        std::unordered_map<instruction_ref, shape> ins_shapes;
        size_t param_cnt               = 0;
        std::vector<std::string> names = mod->get_parameter_names();
        std::sort(names.begin(), names.end());
        for(const std::string& param_name : names)
        {
            ins_shapes[mod->get_parameter(param_name)] = inputs[param_cnt++];
        }
        for(auto ins : iterator_for(*mod))
        {
            if(ins->name() == "@param")
            {
                continue;
            }
            if(ins->name() == "@literal")
            {
                ins_shapes[ins] = ins->get_shape();
                continue;
            }
            if(ins->name() == "@return")
            {
                auto s = ins_shapes[ins->inputs().at(0)].with_type(type);
                if(not s.standard())
                    MIGRAPHX_THROW("MLIR doesnt support non-standard output");
                return s;
            }
            std::vector<shape> input_shapes;
            input_shapes.resize(ins->inputs().size());
            std::transform(ins->inputs().begin(),
                           ins->inputs().end(),
                           input_shapes.begin(),
                           [&](auto in) { return ins_shapes[in]; });
            ins_shapes[ins] = ins->get_operator().compute_shape(input_shapes);
        }
        MIGRAPHX_THROW("No return found in the submodule");
    }
};
MIGRAPHX_REGISTER_OP(mlir_op);

namespace {
std::tuple<instruction_ref, std::vector<instruction_ref>>
fuse_input_ops_and_gemm_based_op(module_ref mm, instruction_ref gemm_based_op)
{
    std::vector<instruction_ref> top_inputs;
    std::vector<instruction_ref> imm_inputs;
    size_t input_cnt = 0;
    for(instruction_ref input : gemm_based_op->inputs())
    {
        std::vector<operation> op_stream;
        while(contains(
            {"slice", "transpose", "contiguous", "reshape", "squeeze", "flatten", "unsqueeze"},
            input->name()))
        {
            operation op = input->get_operator();
            if(contains({"squeeze", "flatten", "unsqueeze"}, input->name()))
            {
                op = migraphx::make_op("reshape", {{"dims", input->get_shape().lens()}});
            }
            op_stream.push_back(op);
            input = input->inputs().at(0);
        }
        top_inputs.push_back(input);
        instruction_ref prev_input =
            mm->add_parameter("y" + std::to_string(input_cnt++), input->get_shape());
        for(const auto& op : reverse(op_stream))
        {
            prev_input = mm->add_instruction(op, {prev_input});
        }
        imm_inputs.push_back(prev_input);
    }
    instruction_ref new_gemm_based_op =
        mm->add_instruction(gemm_based_op->get_operator(), imm_inputs);
    return {new_gemm_based_op, top_inputs};
}

enum class mlir_mode
{
    all,
    fast,
    int8,
    none
};

auto is_mlir_dot(mlir_mode mode)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(mode == mlir_mode::none)
            return false;
        if(ins->name() != "dot" and ins->name() != "quant_dot")
            return false;
        if(mode != mlir_mode::fast)
            return true;
        auto a = ins->inputs().front()->get_shape();
        auto b = ins->inputs().back()->get_shape();
        // auto m = a.lens()[a.lens().size() - 2];
        // auto n = b.lens().back();
        auto k = a.lens().back();
        // Skipping GEMMs with a K dimension greater than 2048 is a course-grained strategy
        // to avoid poor-performing GEMM kernels from MLIR
        // To-do: Investigate a more precise strategy
        return k <= 2048;
    });
}

auto is_mlir_conv(mlir_mode mode)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(mode == mlir_mode::none)
            return false;
        if(ins->name() != "convolution" and ins->name() != "quant_convolution")
            return false;
        value v    = ins->get_operator().to_value();
        auto group = v.at("group").to<int>();
        if(group != 1)
            return false;
        // Avoid MLIR assertion: Index < Length && "Invalid index!"
        if(ins->get_shape().lens().size() != 4)
            return false;
        if(ins->get_shape().type() == shape::int8_type)
            return true;
        if(mode == mlir_mode::int8)
            return false;
        if(mode == mlir_mode::all)
            return true;
        auto w = ins->inputs().at(1)->get_shape();
        if(w.lens().size() != 4)
            return true;
        if(w.lens()[2] != w.lens()[3])
            return true;
        return (w.lens()[3] % 3) != 0;
    });
}

std::unordered_map<instruction_ref, instruction_ref>
    create_param_map_with_literals(module_ref mm, const module* pm, const shape& shape)
    {
        std::unordered_map<instruction_ref, instruction_ref> ins_map;
        for(auto ins : iterator_for(*pm))
        {
            if(ins->name() != "@literal")
            {
                continue;
            }
            literal r               = ins->get_literal();
            instruction_ref literal = mm->add_literal(r);
            instruction_ref mbcast  = mm->add_instruction(
                make_op("multibroadcast", {{"out_lens", shape.lens()}}), literal);
            ins_map[ins] = mbcast;
        }
        return ins_map;
    }

std::vector<instruction_ref> fold_pointwise_mod(instruction_ref pm_ins, module_ref parent_mod, const std::unordered_map<instruction_ref, instruction_ref>& ins_map){
    auto* pm           = pm_ins->module_inputs().front();
    auto names         = pm->get_parameter_names();
    std::sort(names.begin(), names.end());
    std::unordered_map<instruction_ref, instruction_ref> param_map =
            create_param_map_with_literals(parent_mod, pm, pm_ins->get_shape());
    std::transform( names.begin(),
                    names.end(),
                    pm_ins->inputs().begin(),
                    std::inserter(param_map, param_map.end()),
                    [&](auto name, auto input) {
                        if(ins_map.count(input))
                            return std::make_pair(pm->get_parameter(name), ins_map.at(input));
                        return std::make_pair(pm->get_parameter(name),
                                                parent_mod->add_parameter(name, input->get_shape()));
                    });
    return parent_mod->insert_instructions(parent_mod->end(), pm, param_map);
}

// Whitelist supported fusion options, including imposing type constraints
// for cases where MLIR only supports an operation (usually a pointwise function)
// on particular types.
bool is_pointwise_op_supported_by_mlir(const instruction& i)
{
    using type_t                                      = shape::type_t;
    const auto& name                                  = i.name();
    const auto result_type                            = i.get_shape().type();
    const std::initializer_list<type_t> allowed_types = {type_t::float_type,
                                                            type_t::half_type,
                                                            type_t::int8_type,
                                                            type_t::int32_type,
                                                            type_t::bool_type};
    // Preliminary type check.
    if(not contains(allowed_types, result_type))
    {
        return false;
    }
    const std::initializer_list<std::string> any_type_ops = {"@literal", "@param", "@return"};
    const std::initializer_list<std::string> no_bool_ops  = {
        "convolution",
        "quant_convolution",
        "dot",
        "quant_dot",
        "add",
        "clip",
        "relu",
        "sub",
        "mul",
        "div",
        "pow",
        "where",
        "quantizelinear",
        "dequantizelinear",
        "abs",
        "neg",
    };
    const std::initializer_list<std::string> fp_only_ops = {
        "ceil",
        "erf",
        "exp",
        "floor",
        "log",
        "recip",
        "rsqrt",
        "sigmoid",
        "softmax",
        "tanh",
    };
    bool is_float = contains({type_t::float_type, type_t::half_type}, result_type);
    if(contains(any_type_ops, name))
        return true;
    if(result_type != type_t::bool_type and contains(no_bool_ops, name))
        return true;
    if(is_float and contains(fp_only_ops, name))
        return true;
    // Only conversions between floating types are known to be unambigiously
    // supported.
    if(is_float and name == "convert")
    {
        return std::all_of(i.inputs().begin(), i.inputs().end(), [](const auto& arg) {
            return contains({type_t::float_type, type_t::half_type}, arg->get_shape().type());
        });
    }
    return false;
}



struct find_mlir_fused_ops
{
    mlir_mode conv_mode = mlir_mode::none;
    mlir_mode dot_mode  = mlir_mode::none;
    auto matcher() const
    {
        auto dot_or_conv = match::skip(match::name("contiguous"))(
            match::any_of(is_mlir_dot(dot_mode), is_mlir_conv(conv_mode)).bind("gemm_based_op"));
        return match::name("pointwise")(match::any_of[match::inputs()](dot_or_conv.bind("x")));
    }

    void rewrite(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins           = r.result;
        auto gemm_based_op = r.instructions["gemm_based_op"];
        auto x_ins         = r.instructions["x"]; // input after contiguous
        auto* pm           = ins->module_inputs().front();
        auto names         = pm->get_parameter_names();
        // Whitelist pointwise operators.
        if(std::any_of(pm->begin(), pm->end(), [&](const auto& i) {
               return not is_pointwise_op_supported_by_mlir(i);
           }))
            return;

        std::sort(names.begin(), names.end());
        module_ref mm = mpm.create_module("mlir_" + pm->name());
        mm->set_bypass();
        auto [anchor_op, top_inputs] = fuse_input_ops_and_gemm_based_op(mm, gemm_based_op);
        mm->add_return(fold_pointwise_mod(ins, mm, {{x_ins, anchor_op}}));

        std::vector<instruction_ref> inputs;
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != gemm_based_op; });
        inputs.insert(inputs.end(), top_inputs.begin(), top_inputs.end());
        mpm.get_module().replace_instruction(
            ins, mlir_op{gemm_based_op->get_operator()}, inputs, {mm});
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins           = r.result;
        auto* pm           = ins->module_inputs().front();
        // Whitelist pointwise operators.
        if(std::any_of(pm->begin(), pm->end(), [&](const auto& i) {
               return not is_pointwise_op_supported_by_mlir(i);
           }))
            return;
        rewrite(mpm, r);
    }
};

template <auto Matcher>
struct find_mlir_standalone_op
{
    mlir_mode mode = mlir_mode::none;
    auto matcher() const { return Matcher(mode); }

    void rewrite(module_pass_manager& mpm, instruction_ref top_ins) const
    {
      static size_t counter = 0;
      module_ref mm         = mpm.create_module("mlir_" + std::to_string(counter++));
      mm->set_bypass();
      auto [anchor_op, top_inputs] = fuse_input_ops_and_gemm_based_op(mm, top_ins);
      mm->add_return({anchor_op});
      mpm.get_module().replace_instruction(
      top_ins, mlir_op{top_ins->get_operator()}, top_inputs, {mm});
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto conv_based_op = r.result;
        //
        // enable only for fp32/fp16/i8 types
        if(std::any_of(conv_based_op->inputs().begin(), conv_based_op->inputs().end(), [&](auto i) {
               return not contains(
                   {shape::type_t::float_type, shape::type_t::half_type, shape::type_t::int8_type},
                   i->get_shape().type());
           }))
            return;
        rewrite(mpm, conv_based_op);
    }
};

using find_mlir_standalone_convolution_op = find_mlir_standalone_op<&is_mlir_conv>;
using find_mlir_standalone_dot_op         = find_mlir_standalone_op<&is_mlir_dot>;

struct find_mlir_standalone_attention_op
{
    mlir_mode mode = mlir_mode::none;
    void rewrite(module_pass_manager& mpm, const match::matcher_result& r) const
    {
      static size_t counter = 0;
      module_ref mm         = mpm.create_module("mlir_" + std::to_string(counter++));
      std::vector<instruction_ref> inputs;
      mm->set_bypass();

      std::unordered_map<instruction_ref, instruction_ref> ins_map;
      auto top_ins = r.instructions["top_dot"];
      auto [new_top_ins, top_inputs] = fuse_input_ops_and_gemm_based_op(mm, top_ins);
      inputs.insert(inputs.begin(), top_inputs.begin(), top_inputs.end());
      ins_map[top_ins] = new_top_ins;
      if(r.instructions.find("scale") != r.instructions.end()){
        auto scale_ins = r.instructions["scale"];
        new_top_ins = fold_pointwise_mod(scale_ins, mm, ins_map)[0];
        std::copy_if(scale_ins->inputs().begin(),
                     scale_ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != top_ins; });
      }
      auto softmax = mm->add_instruction(r.instructions["softmax"]->get_operator(), new_top_ins);
      std::transform(r.instructions["bottom_dot"]->inputs().begin(),
                     r.instructions["bottom_dot"]->inputs().end(),
                     std::inserter(ins_map, ins_map.end()),
                     [&](auto old_ins) {
                        if(old_ins == r.instructions["softmax"]){
                            return std::make_pair(old_ins, softmax);
                        }
                        inputs.push_back(old_ins);
                        return std::make_pair(old_ins,
                                              mm->add_parameter("bdot_non_smax_in", old_ins->get_shape()));
                     });
      auto bottom_dot_a = ins_map[r.instructions["bottom_dot"]->inputs().front()];
      auto bottom_dot_b = ins_map[r.instructions["bottom_dot"]->inputs().back()];
      auto new_bottom_dot = mm->add_instruction(make_op("dot"), {bottom_dot_a, bottom_dot_b});
      if(r.instructions.find("trailing_pm") != r.instructions.end()){
        new_bottom_dot = fold_pointwise_mod(r.instructions["trailing_pm"], mm, ins_map)[0];
      }
      mm->add_return({new_bottom_dot});
      mpm.get_module().replace_instruction(
      r.instructions["bottom_dot"], mlir_op{new_bottom_dot->get_operator()}, inputs, {mm});
    }

    auto matcher() const {
        auto match_softmax_input =  match::any_of[match::inputs()](match::name("dot").bind("top_dot"), match::name("pointwise")(match::any_of[match::inputs()](match::name("dot").bind("top_dot"))).bind("scale"));
        auto is_mlir_attention = match::name("dot")(match::any_of[match::inputs()](match::name("softmax")(match_softmax_input).bind("softmax"))).bind("bottom_dot");
        return is_mlir_attention;
    }

    bool check(const match::matcher_result& r) const {
        // We are only enabling attention
        // in the highest enablement mode for now
        if(mode != mlir_mode::all){
            return false;
        }
        auto top_dot = r.instructions["top_dot"];
        // Check the pointwise mod only contains a single mul
        if(r.instructions.find("scale") != r.instructions.end()){
            auto scale_pm = r.instructions["scale"];
            bool found_mul = false;
            for(const auto& scale_ins : *scale_pm->module_inputs().front()){
                if(contains({"@param", "@literal", "@return"}, scale_ins.name())){
                    continue;
                }
                if(scale_ins.name() == "mul" && !found_mul){
                    found_mul = true;
                    continue;
                }
                return false;
            }
        }
        // enable only for fp32/fp16/i8 types
        if(std::any_of(top_dot->inputs().begin(), top_dot->inputs().end(), [&](auto i) {
               return not contains(
                   {shape::type_t::float_type, shape::type_t::half_type, shape::type_t::int8_type},
                   i->get_shape().type());
           })){
            return false;
        }
        return true;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        if(!check(r)){
            return;
        }
        rewrite(mpm, r);
    }
};

struct find_mlir_attention_fused_ops : public find_mlir_standalone_attention_op
{
    auto matcher() const {
        auto standalone_matcher = find_mlir_standalone_attention_op::matcher();
        return match::name("pointwise")(match::any_of[match::inputs()](standalone_matcher).bind("trailing_pm"));;
    }

    bool check(const match::matcher_result& r) const {
        if(!find_mlir_standalone_attention_op::check(r)){
            return false;
        }
        auto trailing_pm_ins = r.instructions["trailing_pm"]; // input after contiguous
        auto* trailing_pm = trailing_pm_ins->module_inputs().front();
        // Whitelist pointwise operators.
        if(std::any_of(trailing_pm->begin(), trailing_pm->end(), [&](const auto& i) {
               return not is_pointwise_op_supported_by_mlir(i);
           }))
            return false;
        return true;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        if(!check(r)){
            return;
        }
        rewrite(mpm, r);
    }
};

/**
 * @brief Declares a new MIGraphX environment variable which forces to generate
 * only specific MLIR operations.
 *
 * The variable, if defined, forces MIGraphX to use only specific operations
 * with MLIR regardless of the underlying GPU architecture. The variable accepts
 * a list of operations separated by comma. The variable recognizes the following
 * operations: "fused", "convolution", "dot". If the variable is not defined MIGraphX
 * will decide by itself which operations to delegate to MLIR. The variable is
 * intended to be primarily used by rocMLIR developers.
 */
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_USE_SPECIFIC_OPS);

bool is_requested(std::string_view option, bool fallback = false)
{
    auto string_value  = string_value_of(MIGRAPHX_MLIR_USE_SPECIFIC_OPS{}, "");
    if(string_value.empty())
        return fallback;
    const auto options = split_string(string_value, ',');
    return contains(options, option);
}
} // namespace

#endif // MIGRAPHX_MLIR

void fuse_mlir::apply(module_pass_manager& mpm) const
{
#ifdef MIGRAPHX_MLIR
    const auto& device_name = ctx == nullptr ? "" : ctx->get_current_device().get_gfx_name();
    const bool is_navi      = starts_with(device_name, "gfx110");

    auto get_mode = [&](std::string_view option, mlir_mode m1, mlir_mode m2 = mlir_mode::fast) {
        if(is_requested(option))
            return mlir_mode::all;
        if(is_navi)
            return mlir_mode::all;
        return std::max(m1, m2);
    };

    mlir_mode mode =
        (enabled(MIGRAPHX_ENABLE_EXTRA_MLIR{}) or enable_extra) ? mlir_mode::fast : mlir_mode::none;

    // Attention offloads; default disabled
    match::find_matches(mpm,
                        find_mlir_attention_fused_ops{get_mode("attention", mlir_mode::none)});
    match::find_matches(mpm,
                        find_mlir_standalone_attention_op{get_mode("attention", mlir_mode::none)});

    match::find_matches(mpm,
                        find_mlir_fused_ops{.conv_mode = get_mode("fused", mlir_mode::fast),
                                            .dot_mode  = get_mode("fused", mode)});

    match::find_matches(
        mpm,
        find_mlir_standalone_convolution_op{get_mode("convolution", mlir_mode::int8)},
        find_mlir_standalone_dot_op{get_mode("dot", mlir_mode::none)});
#else
    (void)mpm;
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
