/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/algorithm.hpp>
#include <migraphx/param_utils.hpp>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_EXTRA_MLIR);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MLIR);
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

bool mlir_enabled()
{
#ifdef MIGRAPHX_MLIR
    const bool mlir_disabled = enabled(MIGRAPHX_DISABLE_MLIR{});
    return not mlir_disabled;
#else
    return false;
#endif
}

namespace {
struct requested
{
};
struct rejected
{
};
} // namespace

static bool is_negated_op(const std::string& s)
{
    if(s.empty())
        return false;
    return contains({'!', '~'}, s[0]);
}

template <class Action>
static std::vector<std::string> get_usage()
{
    static const auto options =
        split_string(string_value_of(MIGRAPHX_MLIR_USE_SPECIFIC_OPS{}, ""), ',');
    static const bool enabled = std::is_same<Action, requested>{};
    std::vector<std::string> result;
    auto remove_not_symbol = [&](const std::string& s) {
        if(is_negated_op(s))
            return s.substr(1);
        return s;
    };
    transform_if(
        options.begin(),
        options.end(),
        std::back_inserter(result),
        [&](const std::string& option) {
            if(option.empty())
                return false;
            if(is_negated_op(option))
                return not enabled;
            return enabled;
        },
        remove_not_symbol);
    return result;
}

template <class Action>
static bool specific_op(std::string_view option, bool fallback = false)
{
    static const auto options = get_usage<Action>();
    if(options.empty())
        return fallback;
    if(contains(option, "fused") and contains(options, "fused"))
        return true;
    return contains(options, option);
}

bool mlir_attention_enabled()
{
#ifdef MIGRAPHX_MLIR
    if(not mlir_enabled())
        return false;
    return specific_op<requested>("attention");
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

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        module_ref mod = mods[0];
        check_shapes{inputs, *this}.packed_or_broadcasted();
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");
        if(inputs.size() < 2)
            MIGRAPHX_THROW("should have at least two inputs.");

        auto result =
            mod->compute_shapes(inputs, {.name = name(), .strict_type = true, .strict_lens = true});
        if(result.size() == 1)
            return result.front();
        return shape{result};
    }
};
MIGRAPHX_REGISTER_OP(mlir_op);

namespace {

std::tuple<instruction_ref, std::vector<operation>>
get_fusable_input_op_stream(instruction_ref lower_input)
{
    instruction_ref upper_input = lower_input;
    std::vector<operation> op_stream;
    while(contains({"slice",
                    "transpose",
                    "multibroadcast",
                    "broadcast",
                    "contiguous",
                    "reshape",
                    "squeeze",
                    "flatten",
                    "unsqueeze"},
                   upper_input->name()))
    {
        operation op = upper_input->get_operator();
        if(contains({"squeeze", "flatten", "unsqueeze"}, upper_input->name()))
        {
            op = migraphx::make_op("reshape", {{"dims", upper_input->get_shape().lens()}});
        }
        op_stream.push_back(op);
        upper_input = upper_input->inputs().at(0);
    }
    return {upper_input, op_stream};
}

std::tuple<instruction_ref, std::vector<instruction_ref>>
fuse_input_ops_and_gemm_based_op(module_ref mm,
                                 const std::vector<instruction_ref>& gemm_based_op_inputs,
                                 const operation& gemm_based_op)
{
    std::vector<instruction_ref> top_inputs;
    std::vector<instruction_ref> imm_inputs;
    size_t input_cnt = 0;
    for(instruction_ref input : gemm_based_op_inputs)
    {
        auto [upper_input, op_stream] = get_fusable_input_op_stream(input);
        top_inputs.push_back(upper_input);
        instruction_ref prev_input =
            mm->add_parameter(param_name(input_cnt++, "y"), upper_input->get_shape().as_standard());
        for(const auto& op : reverse(op_stream))
        {
            prev_input = mm->add_instruction(op, {prev_input});
        }
        imm_inputs.push_back(prev_input);
    }
    instruction_ref new_gemm_based_op = mm->add_instruction(gemm_based_op, imm_inputs);
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
        // dot operation where (FP8 * FP8 = FP8) is not available in MLIR. rocBLAS has the support
        // for it.
        if(ins->get_shape().type() == migraphx::shape::fp8e4m3fnuz_type)
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
        return k <= 1024;
    });
}

auto is_mlir_conv(mlir_mode mode)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(mode == mlir_mode::none)
            return false;
        if(ins->name() != "convolution" and ins->name() != "quant_convolution")
            return false;
        auto input = ins->inputs().front()->get_shape();
        value v    = ins->get_operator().to_value();
        auto group = v.at("group").to<int>();
        // Avoid MLIR assertion: Index < Length && "Invalid index!"
#ifdef _WIN32
        // Temporarily make it available only on Windows
        if(ins->get_shape().lens().size() != 4 and group > 1)
            return false;
#else
        if(ins->get_shape().lens().size() != 4)
            return false;
#endif
        if(contains({shape::fp8e4m3fnuz_type, shape::int8_type}, input.type()))
            return true;
        if(mode == mlir_mode::all)
            return true;
        // No windograd for group convolution
        if(group > 1)
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
        instruction_ref mbcast =
            mm->add_instruction(make_op("multibroadcast", {{"out_lens", shape.lens()}}), literal);
        ins_map[ins] = mbcast;
    }
    return ins_map;
}

std::vector<instruction_ref>
fold_pointwise_mod(instruction_ref pm_ins,
                   module_ref parent_mod,
                   const std::unordered_map<instruction_ref, instruction_ref>& ins_map)
{
    auto* pm   = pm_ins->module_inputs().front();
    auto names = pm->get_parameter_names();
    std::sort(names.begin(), names.end());
    std::unordered_map<instruction_ref, instruction_ref> param_map =
        create_param_map_with_literals(parent_mod, pm, pm_ins->get_shape());
    std::transform(names.begin(),
                   names.end(),
                   pm_ins->inputs().begin(),
                   std::inserter(param_map, param_map.end()),
                   [&](auto name, auto input) {
                       if(ins_map.count(input))
                           return std::make_pair(pm->get_parameter(name), ins_map.at(input));
                       return std::make_pair(
                           pm->get_parameter(name),
                           parent_mod->add_parameter(name, input->get_shape().as_standard()));
                   });
    return parent_mod->insert_instructions(parent_mod->end(), pm, &param_map);
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
                                                         type_t::fp8e4m3fnuz_type,
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
    bool is_float =
        contains({type_t::float_type, type_t::half_type, type_t::fp8e4m3fnuz_type}, result_type);
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
        if(result_type == shape::fp8e4m3fnuz_type)
        {
            return false;
        } // else
        return std::all_of(i.inputs().begin(), i.inputs().end(), [](const auto& arg) {
            return contains({type_t::float_type, type_t::half_type}, arg->get_shape().type());
        });
    }
    return false;
}

MIGRAPHX_PRED_MATCHER(mlir_pointwise, instruction_ref ins)
{
    if(ins->name() != "pointwise")
        return false;
    auto* pm = ins->module_inputs().front();
    return std::all_of(pm->begin(), pm->end(), [&](const auto& i) {
        return is_pointwise_op_supported_by_mlir(i);
    });
}

std::vector<instruction_ref> mlir_contiguous(module_pass_manager& mpm,
                                             const std::vector<instruction_ref>& inputs)
{
    std::vector<instruction_ref> result;
    std::transform(
        inputs.begin(), inputs.end(), std::back_inserter(result), [&](instruction_ref input) {
            if(input->get_shape().packed() or input->get_shape().broadcasted())
                return input;
            return mpm.get_module().insert_instruction(
                std::next(input), make_op("contiguous"), input);
        });
    return result;
}

struct find_mlir_fused_ops
{
    mlir_mode conv_mode = mlir_mode::none;
    mlir_mode dot_mode  = mlir_mode::none;
    auto matcher() const
    {
        auto dot_or_conv = match::skip(match::name("contiguous"))(
            match::any_of(is_mlir_dot(dot_mode), is_mlir_conv(conv_mode)).bind("gemm_based_op"));
        return mlir_pointwise()(match::any_of[match::inputs()](dot_or_conv.bind("x")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins           = r.result;
        auto gemm_based_op = r.instructions["gemm_based_op"];
        auto x_ins         = r.instructions["x"]; // input after contiguous
        auto* pm           = ins->module_inputs().front();
        auto names         = pm->get_parameter_names();
        std::sort(names.begin(), names.end());
        module_ref mm = mpm.create_module("mlir_" + pm->name());
        mm->set_bypass();
        auto [anchor_op, top_inputs] = fuse_input_ops_and_gemm_based_op(
            mm, gemm_based_op->inputs(), gemm_based_op->get_operator());
        mm->add_return(fold_pointwise_mod(ins, mm, {{x_ins, anchor_op}}));

        std::vector<instruction_ref> inputs;
        std::copy_if(ins->inputs().begin(),
                     ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != gemm_based_op; });
        inputs.insert(inputs.end(), top_inputs.begin(), top_inputs.end());
        mpm.get_module().replace_instruction(
            ins, mlir_op{gemm_based_op->get_operator()}, mlir_contiguous(mpm, inputs), {mm});
    }
};

template <auto Matcher>
struct find_mlir_standalone_op
{
    mlir_mode mode = mlir_mode::none;
    auto matcher() const { return Matcher(mode); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto gemm_based_op = r.result;
        // enable only for fp32/fp16/i8/fp8 types
        if(std::any_of(gemm_based_op->inputs().begin(), gemm_based_op->inputs().end(), [&](auto i) {
               return not contains({shape::type_t::float_type,
                                    shape::type_t::half_type,
                                    shape::type_t::int8_type,
                                    shape::type_t::fp8e4m3fnuz_type},
                                   i->get_shape().type());
           }))
            return;
        static size_t counter = 0;
        module_ref mm =
            mpm.create_module("mlir_" + gemm_based_op->name() + std::to_string(counter++));
        mm->set_bypass();
        auto [anchor_op, top_inputs] = fuse_input_ops_and_gemm_based_op(
            mm, gemm_based_op->inputs(), gemm_based_op->get_operator());
        mm->add_return({anchor_op});
        mpm.get_module().replace_instruction(gemm_based_op,
                                             mlir_op{gemm_based_op->get_operator()},
                                             mlir_contiguous(mpm, top_inputs),
                                             {mm});
    }
};

using find_mlir_standalone_convolution_op = find_mlir_standalone_op<&is_mlir_conv>;
using find_mlir_standalone_dot_op         = find_mlir_standalone_op<&is_mlir_dot>;

struct find_mlir_standalone_attention_op
{
    auto matcher() const
    {
        return match::name("gpu::pre_gemm_softmax_gemm").bind("gemm_softmax_gemm");
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        static size_t counter  = 0;
        module_ref mm          = mpm.create_module("mlir_" + std::to_string(counter++));
        auto gemm_softmax_gemm = r.instructions["gemm_softmax_gemm"];
        mm->set_bypass();

        auto orig_inputs = gemm_softmax_gemm->inputs();

        std::vector<instruction_ref> gemm0_inputs = {orig_inputs[0], orig_inputs[1]};
        auto [gemm0, top_gemm0_inputs] =
            fuse_input_ops_and_gemm_based_op(mm, gemm0_inputs, make_op("dot"));

        std::vector<instruction_ref> inputs;
        inputs.insert(inputs.begin(), top_gemm0_inputs.begin(), top_gemm0_inputs.end());

        // handle scale
        auto v = gemm_softmax_gemm->get_operator().to_value();
        assert(v.contains("scale"));
        auto scale     = v.at("scale").to<float>();
        auto scale_lit = mm->add_literal(literal{shape{gemm0->get_shape().type()}, {scale}});
        instruction_ref scale_lit_mbcast = mm->add_instruction(
            make_op("multibroadcast", {{"out_lens", gemm0->get_shape().lens()}}), scale_lit);
        auto scaled_gemm0 = mm->add_instruction(make_op("mul"), gemm0, scale_lit_mbcast);

        std::optional<instruction_ref> bias{nullopt};
        if(orig_inputs.size() == 4)
        {
            auto bias_input = orig_inputs[2];
            instruction_ref bias_param =
                mm->add_parameter("y_bias", bias_input->get_shape().as_standard());
            bias = mm->add_instruction(make_op("add"), scaled_gemm0, bias_param);
            inputs.push_back(bias_input);
        }

        auto softmax = mm->add_instruction(
            make_op("softmax", {{"axis", gemm0->get_shape().lens().size() - 1}}),
            bias ? bias.value() : scaled_gemm0);
        auto [old_upper_v, upper_v_op_stream] = get_fusable_input_op_stream(orig_inputs.back());
        instruction_ref new_upper_v =
            mm->add_parameter("z", old_upper_v->get_shape().as_standard());
        for(const auto& op : reverse(upper_v_op_stream))
        {
            new_upper_v = mm->add_instruction(op, {new_upper_v});
        }
        inputs.push_back(old_upper_v);

        auto gemm1 = mm->add_instruction(make_op("dot"), {softmax, new_upper_v});

        std::unordered_map<instruction_ref, instruction_ref> ins_map;
        ins_map[gemm_softmax_gemm] = gemm1;
        auto ins_to_replace        = gemm1;
        auto ins_to_be_replaced    = gemm_softmax_gemm;
        if(r.instructions.find("trailing_pm") != r.instructions.end())
        {
            ins_to_replace = fold_pointwise_mod(r.instructions["trailing_pm"], mm, ins_map)[0];
            std::copy_if(r.instructions["trailing_pm"]->inputs().begin(),
                         r.instructions["trailing_pm"]->inputs().end(),
                         std::back_inserter(inputs),
                         [&](auto input) { return input != gemm_softmax_gemm; });
            ins_to_be_replaced = r.instructions["trailing_pm"];
        }
        mm->add_return({ins_to_replace});

        mpm.get_module().replace_instruction(
            ins_to_be_replaced, mlir_op{gemm1->get_operator()}, mlir_contiguous(mpm, inputs), {mm});
    }
};

struct find_mlir_attention_fused_ops : public find_mlir_standalone_attention_op
{
    auto matcher() const
    {
        auto standalone_matcher = find_mlir_standalone_attention_op::matcher();
        return mlir_pointwise()(
            match::any_of[match::inputs()](standalone_matcher).bind("trailing_pm"));
        ;
    }
};

} // namespace

#endif // MIGRAPHX_MLIR

void fuse_mlir::apply(module_pass_manager& mpm) const
{
#ifdef MIGRAPHX_MLIR
    const auto& device_name = ctx == nullptr ? "" : ctx->get_current_device().get_gfx_name();
    const bool is_navi      = starts_with(device_name, "gfx11");

    auto get_mode = [&](std::string_view option, mlir_mode m1, mlir_mode m2 = mlir_mode::fast) {
        if(specific_op<rejected>(option))
            return mlir_mode::none;
        if(specific_op<requested>(option))
            return mlir_mode::all;
        if(is_navi)
            return mlir_mode::all;
        return std::max(m1, m2);
    };

    // Attention offloads; default disabled
    if(mlir_attention_enabled())
    {
        match::find_matches(mpm, find_mlir_attention_fused_ops{});
        match::find_matches(mpm, find_mlir_standalone_attention_op{});
    }

    match::find_matches(
        mpm,
        find_mlir_fused_ops{.conv_mode = get_mode("fused_convolution", mlir_mode::fast),
                            .dot_mode  = get_mode("fused_dot", mlir_mode::fast)});
    match::find_matches(
        mpm,
        find_mlir_standalone_convolution_op{get_mode("convolution", mlir_mode::fast)},
        find_mlir_standalone_dot_op{get_mode("dot", mlir_mode::fast)});
#else
    (void)mpm;
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
