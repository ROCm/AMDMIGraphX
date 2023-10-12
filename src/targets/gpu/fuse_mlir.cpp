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

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR);

bool mlir_enabled()
{
#ifdef MIGRAPHX_MLIR
    const bool mlir_enabled = enabled(MIGRAPHX_ENABLE_MLIR{});
    if(mlir_enabled)
    {
        return true;
    }
    else
    {

        std::cerr << "WARNING: MIGraphX built with MLIR but it is not enabled. Please set the env "
                     "var MIGRAPHX_ENABLE_MLIR to use MLIR kernel generator."
                  << std::endl;
        return false;
    }
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
        while(contains({"slice", "transpose", "contiguous", "reshape"}, input->name()))
        {
            op_stream.push_back(input->get_operator());
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

MIGRAPHX_PRED_MATCHER(is_mlir_conv, instruction_ref ins)
{
    if(ins->name() != "convolution" and ins->name() != "quant_convolution")
        return false;
    value v    = ins->get_operator().to_value();
    auto group = v.at("group").to<int>();
    if(group != 1)
        return false;
    // Avoid MLIR assertion: Index < Length && "Invalid index!"
    if(ins->get_shape().lens().size() != 4)
        return false;
    return true;
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



struct find_mlir_fused_ops
{
    auto matcher() const
    {
        auto dot_or_conv = match::skip(match::name("contiguous"))(
            match::any_of(match::name("dot"), match::name("quant_dot"), is_mlir_conv())
                .bind("gemm_based_op"));
        return match::name("pointwise")(match::any_of[match::inputs()](dot_or_conv.bind("x")));
    }

    // Whitelist supported fusion options, including imposing type constraints
    // for cases where MLIR only supports an operation (usually a pointwise function)
    // on particular types.
    bool is_pointwise_op_supported_by_mlir(const instruction& i) const
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
        // std::unordered_map<instruction_ref, instruction_ref> param_map =
        //     create_param_map_with_literals(mm, pm, gemm_based_op->get_shape());
        auto [anchor_op, top_inputs] = fuse_input_ops_and_gemm_based_op(mm, gemm_based_op);
        // std::transform(names.begin(),
        //                names.end(),
        //                ins->inputs().begin(),
        //                std::inserter(param_map, param_map.end()),
        //                [&, &anchor = anchor_op](auto name, auto input) {
        //                    if(input == x_ins)
        //                        return std::make_pair(pm->get_parameter(name), anchor);
        //                    return std::make_pair(pm->get_parameter(name),
        //                                          mm->add_parameter(name, input->get_shape()));
        //                });
        // mm->add_return(mm->insert_instructions(mm->end(), pm, param_map));
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

struct find_mlir_attention_fused_ops : public find_mlir_fused_ops
{
    auto matcher() const
    {
        auto match_softmax_input =  match::any_of[match::inputs()](match::name("dot"), match::name("pointwise")(match::any_of[match::inputs()](match::name("dot"))).bind("scale"));
        auto is_mlir_attention = match::name("dot")(match::any_of[match::inputs()](match::name("softmax")));
        return match::name("pointwise")(match::any_of[match::inputs()](is_mlir_attention.bind("x")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins           = r.result;
        auto* pm           = ins->module_inputs().front();
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
                return;
            }
        }
        // Whitelist pointwise operators.
        if(std::any_of(pm->begin(), pm->end(), [&](const auto& i) {
               return not is_pointwise_op_supported_by_mlir(i);
           }))
            return;
        rewrite(mpm, r);
    }
};

struct find_mlir_standalone_op
{
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

struct find_mlir_standalone_convolution_op : find_mlir_standalone_op
{
    auto matcher() const { return is_mlir_conv; }
};

struct find_mlir_standalone_dot_op : find_mlir_standalone_op
{
    auto matcher() const { return match::any_of(match::name("dot"), match::name("quant_dot")); }
};

struct find_mlir_standalone_attention_op : find_mlir_standalone_op
{
    void insert_to_map(std::unordered_map<instruction_ref, instruction_ref>& ins_map, instruction_ref old_ins, instruction_ref new_ins) const {
        if(ins_map.count(new_ins)){
            new_ins = ins_map[new_ins];
        }
        if(!ins_map.count(old_ins)){
            ins_map[old_ins] = new_ins;
        }
    }

    instruction_ref get_from_map(std::unordered_map<instruction_ref, instruction_ref>& ins_map, instruction_ref ins) const {
        if(ins_map.count(ins)){
            return ins_map[ins];
        }
        return ins;
    }

    void rewrite(module_pass_manager& mpm, const match::matcher_result& r) const
    {
      static size_t counter = 0;
      module_ref mm         = mpm.create_module("mlir_" + std::to_string(counter++));
      std::vector<instruction_ref> inputs;
      mm->set_bypass();

      std::unordered_map<instruction_ref, instruction_ref> ins_map;
      auto top_ins = r.instructions["top_dot"];
      auto [new_top_ins, top_inputs] = fuse_input_ops_and_gemm_based_op(mm, top_ins);
      insert_to_map(ins_map, top_ins, new_top_ins);
      if(r.instructions.find("scale") != r.instructions.end()){
        auto scale_ins = r.instructions["scale"];
        new_top_ins = fold_pointwise_mod(scale_ins, mm, ins_map)[0];
        std::copy_if(scale_ins->inputs().begin(),
                     scale_ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != top_ins; });
      }
      auto softmax = mm->add_instruction(r.instructions["softmax"]->get_operator(), new_top_ins);
      insert_to_map(ins_map, r.instructions["softmax"], softmax);
      auto bottom_dot_a = get_from_map(ins_map, r.instructions["bottom_dot"]->inputs().front());
      auto bottom_dot_b = get_from_map(ins_map, r.instructions["bottom_dot"]->inputs().back());
      auto new_bottom_dot = mm->add_instruction(make_op("dot"), {bottom_dot_a, bottom_dot_b});
      mm->add_return({new_bottom_dot});
      inputs.insert(inputs.end(), top_inputs.begin(), top_inputs.end());
      mpm.get_module().replace_instruction(
      top_ins, mlir_op{new_bottom_dot->get_operator()}, inputs, {mm});
    }

    auto matcher() const {
        auto match_softmax_input =  match::any_of[match::inputs()](match::name("dot").bind("top_dot"), match::name("pointwise")(match::any_of[match::inputs()](match::name("dot").bind("top_dot"))).bind("scale"));
        auto is_mlir_attention = match::name("dot")(match::any_of[match::inputs()](match::name("softmax").bind("softmax"))).bind("bottom_dot");
        return is_mlir_attention;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto top_dot = r.instructions["top_dot"];
        // Check the pointwise mod only contains a single mul
        std::cerr << "standalone attention found!\n";
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
                std::cerr << "standalone attention scale not compatible!\n";
                return;
            }
        }
        // enable only for fp32/fp16/i8 types
        if(std::any_of(top_dot->inputs().begin(), top_dot->inputs().end(), [&](auto i) {
               return not contains(
                   {shape::type_t::float_type, shape::type_t::half_type, shape::type_t::int8_type},
                   i->get_shape().type());
           })){
            std::cerr << "standalone attention dtype not compatible!\n";
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
bool is_self_decide() { return string_value_of(MIGRAPHX_MLIR_USE_SPECIFIC_OPS{}, "").empty(); }

bool is_requested(std::string_view option)
{
    assert(not is_self_decide());
    auto string_value  = string_value_of(MIGRAPHX_MLIR_USE_SPECIFIC_OPS{}, "");
    const auto options = split_string(string_value, ',');
    return contains(options, option);
}

bool is_enabled(std::string_view op_name, context* ctx)
{
    if(is_self_decide())
    {
        if(op_name == "fused")
        {
            return true;
        }
        else if(op_name == "convolution" or op_name == "quant_convolution")
        {
            if(ctx == nullptr)
            {
                return false;
            }
            else
            {
                const auto& device = ctx->get_current_device();
                const std::string navi_family{"gfx110"};
                return starts_with(device.get_gfx_name(), navi_family);
            }
        }
        else
        {
            return false;
        }
    }
    return is_requested(op_name);
}
} // namespace

#endif // MIGRAPHX_MLIR

void fuse_mlir::apply(module_pass_manager& mpm) const
{
#ifdef MIGRAPHX_MLIR
    if(is_enabled("fused", this->ctx))
    {   
        match::find_matches(mpm, find_mlir_attention_fused_ops{});
        match::find_matches(mpm, find_mlir_fused_ops{});
    }
    match::find_matches(mpm, find_mlir_standalone_attention_op{});

    if(is_enabled("convolution", this->ctx))
    {
        match::find_matches(mpm, find_mlir_standalone_convolution_op{});
    }

    if(is_enabled("dot", this->ctx))
    {
        match::find_matches(mpm, find_mlir_standalone_dot_op{});
    }
#else
    (void)mpm;
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
