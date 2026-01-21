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
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/env.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/common.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/output_iterator.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/match/softmax.hpp>
#include <migraphx/fp8_types.hpp>
#include <optional>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_EXTRA_MLIR);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MLIR_GEG_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_CEG_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_FLASH_DECODING_ENABLED);
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

bool mlir_attention_enabled(context* ctx)
{
#ifdef MIGRAPHX_MLIR
    if(not mlir_enabled())
        return false;
    if(specific_op<rejected>("attention"))
        return false;
    // Enable attention by default for mi300
    if(ctx != nullptr and starts_with(ctx->get_current_device().get_gfx_name(), "gfx94"))
        return true;
    return specific_op<requested>("attention");
#else
    return false;
#endif
}

bool mlir_flash_decoding_enabled()
{
#ifdef MIGRAPHX_MLIR
    if(not mlir_enabled())
        return false;

    // Check if explicitly enabled via environment variable
    if(enabled(MIGRAPHX_FLASH_DECODING_ENABLED{}))
        return true;

    return false;
#else
    // Without MLIR, only enable if explicitly requested via env var
    return enabled(MIGRAPHX_FLASH_DECODING_ENABLED{});
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

    // Check if the shape can be created from a transpose/broadcast/slice
    static bool is_mlir_compatible(const shape& s)
    {
        if(s.standard() or s.packed() or s.scalar() or s.ndim() == 1)
            return true;
        auto ns = reorder_shape(s, find_permutation(s));
        std::vector<std::size_t> stride_ratios;
        auto last = std::find(ns.strides().begin(), ns.strides().end(), 0);
        if(*std::prev(last) != 1)
            return false;
        std::adjacent_difference(ns.strides().begin(),
                                 last,
                                 std::back_inserter(stride_ratios),
                                 [](auto y, auto x) -> std::size_t {
                                     assert(y != 0);
                                     if((x % y) != 0)
                                         return 0;
                                     return x / y;
                                 });
        return std::equal(stride_ratios.begin() + 1,
                          stride_ratios.end(),
                          ns.lens().begin() + 1,
                          [](auto ratio, auto len) { return ratio >= len; });
    }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        module_ref mod = mods[0];
        check_shapes{inputs, *this}.has_at_least(1);
        if(mods.size() != 1)
            MIGRAPHX_THROW("should have one submodule.");

        if(not std::all_of(inputs.begin(), inputs.end(), &is_mlir_compatible))
            MIGRAPHX_THROW("Shape is not mlir compatible.");

        auto result =
            mod->compute_shapes(inputs, {.name = name(), .strict_type = true, .strict_lens = true});
        if(result.size() == 1)
            return result.front();
        return shape{result};
    }
};
MIGRAPHX_REGISTER_OP(mlir_op);

namespace {

const auto& reshaper_names()
{
    // clang-format off
    static const std::unordered_set<std::string> names = {
        "transpose",
        "multibroadcast",
        "broadcast",
        "contiguous",
        "reshape",
        "reshape_lazy",
        "squeeze",
        "flatten",
        "unsqueeze"
    };
    // clang-format on
    return names;
}

bool is_fusable_input_op(const std::string& name)
{
    return contains(reshaper_names(), name) or contains({"slice"}, name);
}

std::tuple<instruction_ref, std::vector<operation>>
get_fusable_input_op_stream(instruction_ref lower_input)
{
    instruction_ref upper_input = lower_input;
    std::vector<operation> op_stream;
    while(is_fusable_input_op(upper_input->name()))
    {
        operation op = upper_input->get_operator();
        op_stream.push_back(op);
        upper_input = upper_input->inputs().at(0);
    }
    return {upper_input, op_stream};
}

void fuse_input_ops(module_ref mm,
                    const std::vector<instruction_ref>& inputs,
                    std::unordered_map<instruction_ref, instruction_ref>* map_ins)
{
    assert(map_ins != nullptr);
    size_t input_cnt = mm->get_parameters().size();
    for(instruction_ref input : inputs)
    {
        if(contains(*map_ins, input))
            continue;
        auto [upper_input, op_stream] = get_fusable_input_op_stream(input);
        if(not contains(*map_ins, upper_input))
            (*map_ins)[upper_input] =
                mm->add_parameter(param_name(input_cnt++), upper_input->get_shape().as_standard());
        instruction_ref prev_input = (*map_ins)[upper_input];
        for(const auto& op : reverse(op_stream))
        {
            prev_input = mm->add_instruction(op, {prev_input});
        }
        (*map_ins)[input] = prev_input;
    }
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
        // dot operation where (FP8 * FP8 = FP8) is not available in MLIR. rocBLAS/hipBLASLt should
        // have the support for it.
        if(contains(fp8_types{}.get(), ins->get_shape().type()))
            return false;
        // MX types quantization has 4 inputs
        // having all MX GEMM go to MLIR
        if(ins->inputs().size() == 4)
        {
            return true;
        }
        if(mode != mlir_mode::fast)
            return true;
        auto a = ins->inputs().front()->get_shape();
        auto b = ins->inputs().back()->get_shape();
        auto g = std::accumulate(a.lens().begin(), a.lens().end() - 2, 1, std::multiplies<>{});
        auto m = a.lens()[a.lens().size() - 2];
        auto n = b.lens().back();
        auto k = a.lens().back();
        // Skipping GEMMs with a K dimension greater than 2048 is a course-grained strategy
        // to avoid poor-performing GEMM kernels from MLIR
        // TODO: Investigate a more precise strategy
        if(k > 1535)
            return false;
        if(k < 1024)
            return true;
        return (g * m * n) < (384 * 384);
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
        if(ins->get_shape().lens().size() != 4 and group > 1)
            return false;
        std::set<shape::type_t> supported_types = fp8_types{}.get();
        supported_types.insert(shape::int8_type);
        if(contains(supported_types, input.type()))
            return true;
        if(mode == mlir_mode::all)
            return true;
        // No winograd for group convolution
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

auto is_mlir_conv_backwards(mlir_mode mode)
{
    return match::make_basic_pred_matcher([=](instruction_ref ins) {
        if(mode == mlir_mode::none)
            return false;

        if(ins->name() != "convolution_backwards")
            return false;

        auto input = ins->inputs().front()->get_shape();
        if(not contains(
               {shape::type_t::float_type, shape::type_t::half_type, shape::type_t::bf16_type},
               input.type()))
            return false;

        auto w = ins->inputs().at(1)->get_shape();
        // currently handle on 2D conv_backwards in MLIR
        if(w.lens().size() != 4)
            return false;

        value v    = ins->get_operator().to_value();
        auto group = v.at("group").to<int>();
        // currently handle only group == 1
        return (group == 1);
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

instruction_ref insert_pointwise(module& m,
                                 instruction_ref ins,
                                 const operation& op,
                                 const std::vector<instruction_ref>& inputs,
                                 const std::vector<module_ref>& mod_args)
{
    // Only used in assert
    (void)mod_args;
    assert(mod_args.empty());
    return insert_common_op(m, ins, op, inputs, {.common_type = false});
}

instruction_ref unroll_pointwise(module& main_mod,
                                 instruction_ref pos,
                                 const operation& op,
                                 const std::vector<instruction_ref>& inputs,
                                 const std::vector<module_ref>& mod_args)
{
    if(op.name() == "pointwise")
    {
        auto* sub_pm     = mod_args.front();
        auto param_map_2 = create_param_map_with_literals(
            &main_mod, sub_pm, op.compute_shape(to_shapes(inputs), mod_args));
        return main_mod.insert_inline(pos, *sub_pm, inputs, &param_map_2)
            .front(); // cppcheck-suppress returnDanglingLifetime;
    }
    return main_mod.insert_instruction(pos, op, inputs, mod_args);
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
                                                         type_t::bf16_type,
                                                         type_t::half_type,
                                                         type_t::fp8e4m3fnuz_type,
                                                         type_t::fp8e5m2fnuz_type,
                                                         type_t::fp8e4m3fn_type,
                                                         type_t::fp8e5m2_type,
                                                         type_t::int8_type,
                                                         type_t::uint8_type,
                                                         type_t::int32_type,
                                                         type_t::uint32_type,
                                                         type_t::bool_type};
    // Preliminary type check.
    if(not contains(allowed_types, result_type))
    {
        return false;
    }
    const std::initializer_list<std::string> any_type_ops = {"@literal", "@param", "@return"};
    const std::initializer_list<std::string> no_bool_ops  = {
        "abs",
        "add",
        "clip",
        "convolution",
        "dequantizelinear",
        "div",
        "dot",
        "leaky_relu",
        "mul",
        "neg",
        "pow",
        "quant_convolution",
        "quant_dot",
        "quantizelinear",
        "relu",
        "sub",
        "where",
    };
    const std::initializer_list<std::string> fp_only_ops = {
        "ceil",
        "erf",
        "exp",
        "floor",
        "log",
        "recip",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "softmax",
        "tanh",
    };
    std::set<shape::type_t> float_types = {type_t::float_type,
                                           type_t::half_type,
                                           type_t::bf16_type,
                                           type_t::fp8e4m3fnuz_type,
                                           type_t::fp8e5m2fnuz_type,
                                           type_t::fp8e4m3fn_type,
                                           type_t::fp8e5m2_type};
    bool is_float                       = contains(float_types, result_type);
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
        if(contains(fp8_types{}.get(), result_type))
        {
            return false;
        } // else
        return std::all_of(i.inputs().begin(), i.inputs().end(), [](const auto& arg) {
            return contains({type_t::float_type, type_t::half_type, type_t::bf16_type},
                            arg->get_shape().type());
        });
    }
    return false;
}

bool is_reduce_op_supported_by_mlir(const instruction& i)
{
    using type_t                                      = shape::type_t;
    const auto& name                                  = i.name();
    const auto result_type                            = i.get_shape().type();
    const std::initializer_list<type_t> allowed_types = {type_t::float_type,
                                                         type_t::half_type,
                                                         type_t::bf16_type,
                                                         type_t::fp8e4m3fnuz_type,
                                                         type_t::fp8e5m2fnuz_type,
                                                         type_t::fp8e4m3fn_type,
                                                         type_t::fp8e5m2_type};

    // Preliminary type check.
    if(not contains(allowed_types, result_type))
    {
        return false;
    }
    const std::initializer_list<std::string> reduce_ops = {"reduce_mean", "reduce_sum"};
    return contains(reduce_ops, i.name());
}

// A separate function so we can remove operators that are supported by mlir
// but not supported for an input fusion.
bool is_pointwise_op_supported_by_mlir_for_input(const instruction& i)
{
    return is_pointwise_op_supported_by_mlir(i);
}

bool is_reduce(const instruction& ins) { return contains(ins.name(), "reduce"); }

MIGRAPHX_PRED_MATCHER(mlir_split_reduce, instruction_ref ins)
{
    if(ins->name() != "split_fused_reduce")
        return false;
    auto* mod_arg                            = ins->module_inputs().front();
    std::unordered_set<std::string> builtins = {"@param", "@literal", "@return"};
    for(const auto i : iterator_for(*mod_arg))
    {
        if(is_reduce(*i))
        {
            if(not is_reduce_op_supported_by_mlir(*i))
                return false;
        }
        else if(i->name() == "pointwise")
        {
            if(not std::all_of(i->module_inputs().front()->begin(),
                               i->module_inputs().front()->end(),
                               &is_pointwise_op_supported_by_mlir))
                return false;
        }
        else if(not contains(reshaper_names(), i->name()) and not contains(builtins, i->name()))
        {
            return false;
        }
    }
    return true;
}

MIGRAPHX_PRED_MATCHER(mlir_pointwise, instruction_ref ins)
{
    if(ins->name() != "pointwise")
        return false;
    auto* pm = ins->module_inputs().front();
    return std::all_of(pm->begin(), pm->end(), &is_pointwise_op_supported_by_mlir);
}

MIGRAPHX_PRED_MATCHER(mlir_input_pointwise, instruction_ref ins)
{
    if(ins->name() != "pointwise")
        return false;
    auto* pm = ins->module_inputs().front();
    return std::all_of(pm->begin(), pm->end(), &is_pointwise_op_supported_by_mlir_for_input);
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

struct find_mlir_split_reduce
{
    mlir_mode conv_mode = mlir_mode::none;
    mlir_mode dot_mode  = mlir_mode::none;
    auto matcher() const
    {
        auto dot_or_conv = match::name("gpu::mlir_op");
        // TODO: Handle reshapes inbetween
        return mlir_split_reduce()(match::any_of[match::inputs()](dot_or_conv.bind("gemm")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto reduce_ins = r.result;
        auto gemm_ins   = r.instructions["gemm"];
        assert(gemm_ins->get_shape().sub_shapes().empty());
        auto* rm   = reduce_ins->module_inputs().front();
        auto names = rm->get_parameter_names();
        std::sort(names.begin(), names.end());
        module_ref gemm_old_mm = gemm_ins->module_inputs().front();
        module_ref mm = mpm.create_module(gemm_old_mm->name() + "_" + rm->name(), *gemm_old_mm);
        // remove last return instruction
        if(std::prev(mm->end())->name() == "@return")
        {
            mm->remove_instruction(std::prev(mm->end()));
        }
        mm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> param_map;
        param_map[gemm_ins]      = std::prev(mm->end());
        bool gemm_has_multi_outs = gemm_ins->outputs().size() > 1;
        auto return_vals = mm->fuse(*rm, reduce_ins->inputs(), &param_map, &unroll_pointwise);
        if(gemm_has_multi_outs)
        {
            return_vals.insert(return_vals.end(), param_map[gemm_ins]);
        }
        mm->add_return(return_vals);
        std::vector<instruction_ref> inputs;
        std::copy_if(reduce_ins->inputs().begin(),
                     reduce_ins->inputs().end(),
                     std::back_inserter(inputs),
                     [&](auto input) { return input != gemm_ins; });
        inputs.insert(inputs.end(), gemm_ins->inputs().begin(), gemm_ins->inputs().end());
        if(gemm_has_multi_outs)
        {
            auto fused_ins = mpm.get_module().insert_instruction(
                reduce_ins, mlir_op{gemm_ins->get_operator()}, mlir_contiguous(mpm, inputs), {mm});
            auto dot_ins = mpm.get_module().insert_instruction(
                reduce_ins,
                migraphx::make_op("get_tuple_elem", {{"index", return_vals.size() - 1}}),
                fused_ins);

            mpm.get_module().replace_instruction(gemm_ins, dot_ins);
            for(const auto& outs : reduce_ins->outputs())
            {
                assert(outs->get_operator().name() == "get_tuple_elem");
                mpm.get_module().replace_instruction(outs, outs->get_operator(), fused_ins);
            }
        }
        else
        {
            mpm.get_module().replace_instruction(
                reduce_ins, mlir_op{gemm_ins->get_operator()}, mlir_contiguous(mpm, inputs), {mm});
        }
    }
};

/**
 * Fuses rocMLIR compatible dot or conv op -> reshapes -> pointwise
 * into a mlir_op with submodule.
 */
struct find_mlir_fused_ops
{
    mlir_mode conv_mode = mlir_mode::none;
    mlir_mode dot_mode  = mlir_mode::none;

    static auto make_conv_dot_reshaper_names()
    {
        auto names = reshaper_names();
        names.erase("broadcast");
        names.erase("multibroadcast");
        return names;
    }

    /**
     * Matches:
     * mlir_dot_or_conv <binds to "gemm_based_op"> ->
     * skip(conv_dot_reshaper_names) <binds to "x"> ->
     * mlir_pointwise <matcher result>
     */
    auto matcher() const
    {
        static const auto conv_dot_reshaper_names = make_conv_dot_reshaper_names();
        auto dot_or_conv = match::skip(match::name(conv_dot_reshaper_names))(
            match::any_of(is_mlir_dot(dot_mode), is_mlir_conv(conv_mode)).bind("gemm_based_op"));
        return mlir_pointwise()(match::any_of[match::inputs()](dot_or_conv.bind("x")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto pw_ins        = r.result;
        auto gemm_based_op = r.instructions["gemm_based_op"];
        auto x_ins         = r.instructions["x"]; // input to pointwise after reshaper op stream
        auto* pm           = pw_ins->module_inputs().front();
        auto pw_inputs     = pw_ins->inputs();
        // only of one of the inputs to pointwise module should be dependent on conv/gemm that is
        // being fused, otherwise it can create invalid graph transformation
        if(std::any_of(pw_inputs.begin(), pw_inputs.end(), [&](const auto& i) {
               return i != x_ins and reaches(gemm_based_op, i);
           }))
            return;

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        module_ref mm = mpm.create_module("mlir_" + pm->name());
        mm->set_bypass();
        fuse_input_ops(mm, gemm_based_op->inputs(), &map_ins);

        bool gemm_has_multi_outs = gemm_based_op->outputs().size() > 1;
        std::vector<instruction_ref> inss_to_insert;
        auto reshape_ins = x_ins;
        for(; reshape_ins != gemm_based_op; reshape_ins = reshape_ins->inputs().front())
        {
            inss_to_insert.push_back(reshape_ins);
            gemm_has_multi_outs |= reshape_ins->outputs().size() > 1;
        }
        inss_to_insert.push_back(gemm_based_op);

        // Set tag if there are no reshape ops between gemm and pointwise
        if(x_ins == gemm_based_op)
        {
            mm->set_tag("gemm_pointwise");
        }
        std::reverse(inss_to_insert.begin(), inss_to_insert.end());
        mm->add_instructions(inss_to_insert, &map_ins);

        fuse_input_ops(mm, pw_ins->inputs(), &map_ins);
        auto rins = mm->fuse(*pm, pw_ins->inputs(), &map_ins, &insert_pointwise);
        if(gemm_has_multi_outs)
        {
            rins.push_back(map_ins.at(gemm_based_op));
        }
        mm->add_return(rins);

        auto inputs    = find_inputs(map_ins, &mpm.get_module(), mm);
        auto fused_ins = mpm.get_module().insert_instruction(
            pw_ins, mlir_op{gemm_based_op->get_operator()}, mlir_contiguous(mpm, inputs), {mm});
        if(gemm_has_multi_outs)
        {
            auto dot_ins = mpm.get_module().insert_instruction(
                pw_ins,
                migraphx::make_op("get_tuple_elem", {{"index", rins.size() - 1}}),
                fused_ins);

            // move all the reshape instructions after the fused op to avoid
            // generating invalid migraphx program since the reshapes can be
            // used by the replaced dot_ins
            for(instruction_ref x : inss_to_insert)
            {
                if(x == gemm_based_op)
                    continue;
                mpm.get_module().move_instruction(x, pw_ins);
            }

            mpm.get_module().replace_instruction(gemm_based_op, dot_ins);
            if(rins.size() == 2)
            {
                mpm.get_module().replace_instruction(
                    pw_ins, migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused_ins);
            }
        }
        else
        {
            mpm.get_module().replace_instruction(pw_ins, fused_ins);
        }
    }
};

/**
 * Fuses rocMLIR conv/dot -> optional pointwise -> dot -> optional pointwise chain
 * into a mlir_op with submodule.
 */
struct find_mlir_fused_geg_ops
{
    mlir_mode conv_mode = mlir_mode::none;
    mlir_mode dot_mode  = mlir_mode::none;
    std::string gfx_name;

    // check if individual GEMM meets architecture requirements
    bool is_gemm_supported(instruction_ref ins) const
    {
        // on navis, wmma doesn't support fp32, so skip fp32 GEMMs
        // one gemm being f32 is sufficient to turn off this fusion
        if(starts_with(gfx_name, "gfx11") or starts_with(gfx_name, "gfx12"))
        {
            if(ins->get_shape().type() == shape::type_t::float_type)
            {
                return false;
            }
        }
        return true;
    }

    // Check if instruction is a gpu::mlir_op with a tag indicating standalone or gemm_pointwise
    // fusion
    static bool is_mlir_fusible_gemm_op(instruction_ref ins)
    {
        if(ins->name() != "gpu::mlir_op")
            return false;

        if(ins->module_inputs().empty())
            return false;

        auto tag = ins->module_inputs().front()->get_tag();
        return tag == "standalone_dot" or tag == "standalone_convolution" or
               tag == "gemm_pointwise";
    }

    // check heuristic based on the 4 GEMM inputs to determine if fusion should proceed
    bool check_heuristic(const std::vector<instruction_ref>& first_inputs,
                         const std::vector<instruction_ref>& second_inputs) const
    {
        auto shape_a1 = first_inputs.front()->get_shape();
        auto shape_b1 = first_inputs.back()->get_shape();
        auto shape_a2 = second_inputs.front()->get_shape();
        auto shape_b2 = second_inputs.back()->get_shape();

        // gemm0: (m x k) @ (k x n) -> (m x n)
        // gemm1: (m x n) @ (n x g) -> (m x g)
        std::int64_t m = shape_a1.lens()[shape_a1.lens().size() - 2];
        std::int64_t n = shape_b1.lens().back();
        std::int64_t k = shape_a1.lens().back();
        std::int64_t g = shape_b2.lens().back();

        // skip if any dimension is 0 (extraction failed)
        if(m == 0 or n == 0 or k == 0 or g == 0)
            return false;

        // Experimental results show fusion is beneficial when:
        // 1. m is relatively larger than other dimensions (avg difference > 1000)
        // 2. n is relatively larger than other dimensions (avg difference > 1000)
        // Fusion is detrimental when k or g are relatively large

        const double threshold = 1000.0;

        // calculate average differences for m being large
        double m_minus_n        = m - n;
        double m_minus_k        = m - k;
        double m_minus_g        = m - g;
        double m_avg_difference = (m_minus_n + m_minus_k + m_minus_g) / 3.0;

        // calculate average differences for n being large
        double n_minus_m        = n - m;
        double n_minus_k        = n - k;
        double n_minus_g        = n - g;
        double n_avg_difference = (n_minus_m + n_minus_k + n_minus_g) / 3.0;

        // calculate average differences for k being large (detrimental case)
        double k_minus_m        = k - m;
        double k_minus_n        = k - n;
        double k_minus_g        = k - g;
        double k_avg_difference = (k_minus_m + k_minus_n + k_minus_g) / 3.0;

        // calculate average differences for g being large (detrimental case)
        double g_minus_m        = g - m;
        double g_minus_n        = g - n;
        double g_minus_k        = g - k;
        double g_avg_difference = (g_minus_m + g_minus_n + g_minus_k) / 3.0;

        // fusion is good if m or n is significantly larger than others
        if(m_avg_difference > threshold or n_avg_difference > threshold)
            return true;

        // fusion is bad if k or g is significantly larger than others
        if(k_avg_difference > threshold or g_avg_difference > threshold)
            return false;

        // default to not fusing
        return false;
    }

    /*
     * Matches:
     * gpu::mlir_op(standalone_dot/standalone_convolution/gemm_pointwise) <binds to
     * "first_gemm_based_op"> -> gpu::mlir_op(standalone_dot/standalone_convolution/gemm_pointwise)
     * <matcher result, binds to "second_gemm_op">
     */
    auto matcher() const
    {
        auto mlir_gemm_op = match::make_basic_pred_matcher(&is_mlir_fusible_gemm_op);

        auto first_mlir_op = mlir_gemm_op.bind("first_gemm_based_op");
        // this is specifically disallowing multi out from the first submodule, which rocMLIR does
        // not support
        return mlir_gemm_op(match::arg(0)(first_mlir_op)).bind("second_gemm_op");
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto second_gemm_ins = r.result;
        auto first_gemm_ins  = r.instructions["first_gemm_based_op"];

        // extract the actual gemm instructions from the submodules
        auto* first_submod = first_gemm_ins->module_inputs().front();
        auto first_gemm =
            std::find_if(first_submod->begin(), first_submod->end(), [](const auto& i) {
                return contains({"dot", "convolution"}, i.name());
            });

        auto* second_submod = second_gemm_ins->module_inputs().front();
        auto second_gemm    = std::find_if(second_submod->begin(),
                                        second_submod->end(),
                                        [](const auto& i) { return contains({"dot"}, i.name()); });

        // check if both gemms exist and are supported on this architecture
        if(first_gemm == first_submod->end() or second_gemm == second_submod->end())
            return;
        if(not is_gemm_supported(first_gemm) or not is_gemm_supported(second_gemm))
            return;

        // TODO
        // only one input to second_gemm should depend on first_gemm
        // auto second_gemm_inputs = second_gemm_ins->inputs();
        // if(std::any_of(second_gemm_inputs.begin(), second_gemm_inputs.end(), [&](const auto& i) {
        //        return i != first_gemm_ins and reaches(first_gemm_ins, i);
        //    }))
        //     return;

        // check heuristic to determine if fusion should proceed
        if(not check_heuristic(first_gemm->inputs(), second_gemm->inputs()))
            return;

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        std::string module_name = first_submod->name() + "_" + second_submod->name() + "_geg";
        module_ref mm           = mpm.create_module(module_name);
        mm->set_bypass();
        fuse_input_ops(mm, first_gemm_ins->inputs(), &map_ins);

        // check if the first submodule's output is used multiple times
        // the first submodule always has exactly one output, since we do not match on
        // get_tuple_elem however, the result of the pointwise operation may have multiple outputs
        if(first_gemm_ins->outputs().size() > 1)
            return;

        // check if the second submodule has intermediates returned
        bool second_gemm_has_multi_outs = second_gemm_ins->outputs().size() > 1;

        // fuse the first submodule into the new module
        fuse_input_ops(mm, first_gemm_ins->inputs(), &map_ins);
        auto first_rins = mm->fuse(*first_submod, first_gemm_ins->inputs(), &map_ins);
        // first_rins has exactly one element (the single output of the first submodule)
        map_ins[first_gemm_ins] = first_rins.front();

        // fuse the second submodule into the new module
        fuse_input_ops(mm, second_gemm_ins->inputs(), &map_ins);
        auto second_rins = mm->fuse(*second_submod, second_gemm_ins->inputs(), &map_ins);
        // second_rins contains the outputs of the second submodule (may be multiple if
        // gemm_pointwise with multi-outs)
        map_ins[second_gemm_ins] = second_rins.front();

        // if second submodule has multi-outs, second_rins already contains [pointwise, gemm]
        // otherwise it just contains [final_output]
        mm->add_return(second_rins);
        auto inputs = find_inputs(map_ins, &mpm.get_module(), mm);

        // replace the second gemm with the fused GEG operation
        auto fused_ins =
            mpm.get_module().insert_instruction(second_gemm_ins,
                                                mlir_op{second_rins.front()->get_operator()},
                                                mlir_contiguous(mpm, inputs),
                                                {mm});
        if(second_gemm_has_multi_outs)
        {
            // the second submodule returns multiple values: [pointwise_result, gemm_result]
            // find any get_tuple_elem instructions that extract from second_gemm_ins and update
            // them
            for(auto out : second_gemm_ins->outputs())
            {
                if(out->name() == "get_tuple_elem")
                {
                    // update the get_tuple_elem to point to the new fused instruction
                    mpm.get_module().replace_instruction(out, out->get_operator(), fused_ins);
                }
            }

            // replace the pointwise result (the main output) with tuple elem 0
            if(second_rins.size() == 2)
            {
                mpm.get_module().replace_instruction(
                    second_gemm_ins,
                    migraphx::make_op("get_tuple_elem", {{"index", 0}}),
                    fused_ins);
            }
        }
        else
        {
            mpm.get_module().replace_instruction(second_gemm_ins, fused_ins);
        }
    }
};

template <auto Matcher>
struct find_mlir_standalone_op
{
    mlir_mode mode       = mlir_mode::none;
    std::size_t* counter = nullptr;
    auto matcher() const { return Matcher(mode); }

    std::string get_count() const
    {
        if(counter == nullptr)
            MIGRAPHX_THROW("Invalid counter");
        return std::to_string((*counter)++);
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto gemm_based_op = r.result;
        // enable only for fp32/fp16/i8/fp8 types
        if(std::any_of(gemm_based_op->inputs().begin(), gemm_based_op->inputs().end(), [&](auto i) {
               return not contains({shape::type_t::float_type,
                                    shape::type_t::half_type,
                                    shape::type_t::bf16_type,
                                    shape::type_t::int8_type,
                                    shape::type_t::fp8e4m3fnuz_type,
                                    shape::type_t::fp8e5m2fnuz_type,
                                    shape::type_t::fp8e4m3fn_type,
                                    shape::type_t::fp8e5m2_type},
                                   i->get_shape().type());
           }))
            return;
        std::string module_name = "mlir_" + gemm_based_op->name() + get_count();
        if(mpm.get_module().name() != "main")
            module_name = mpm.get_module().name() + ":" + module_name;
        module_ref mm = mpm.create_module(module_name);
        mm->set_bypass();
        mm->set_tag("standalone_" + gemm_based_op->name());
        auto [anchor_op, top_inputs] = fuse_input_ops_and_gemm_based_op(
            mm, gemm_based_op->inputs(), gemm_based_op->get_operator());
        mm->add_return({anchor_op});
        mpm.get_module().replace_instruction(gemm_based_op,
                                             mlir_op{gemm_based_op->get_operator()},
                                             mlir_contiguous(mpm, top_inputs),
                                             {mm});
    }
};

using find_mlir_standalone_conv_backwards_op = find_mlir_standalone_op<&is_mlir_conv_backwards>;
using find_mlir_standalone_conv_op           = find_mlir_standalone_op<&is_mlir_conv>;
using find_mlir_standalone_dot_op            = find_mlir_standalone_op<&is_mlir_dot>;

struct find_mlir_kv_cache_attention_op
{
    mlir_mode dot_mode = mlir_mode::none;

    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "kv_cache_attention"));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto group   = r.result;
        auto* m_attn = group->module_inputs()[0];
        mpm.get_module().replace_instruction(
            group, mlir_op{group->get_operator()}, mlir_contiguous(mpm, group->inputs()), {m_attn});
    }
};

struct find_mlir_attention_op
{
    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", "attention")).bind("group");
    }

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

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto group = r.instructions["group"];

        auto* group_mod = group->module_inputs().front();

        std::string module_name = "mlir_" + group_mod->name();
        module_ref mlir_attn    = mpm.create_module(module_name);
        mlir_attn->set_bypass();

        // Fuse any input reshapes
        std::unordered_map<instruction_ref, instruction_ref> map_main_to_mlir_attn;
        fuse_input_ops(mlir_attn, group->inputs(), &map_main_to_mlir_attn);

        std::unordered_map<instruction_ref, instruction_ref> map_group_mod_to_mlir_attn(
            map_main_to_mlir_attn);
        auto attn_outs = mlir_attn->fuse(*group_mod, group->inputs(), &map_group_mod_to_mlir_attn);

        std::unordered_map<instruction_ref, instruction_ref> inss_to_replace;
        // Only fuse attention outputs that are used once by pointwise ops
        if(group->outputs().size() == 1 or attn_outs.size() > 1)
        {
            std::size_t out_idx  = 0;
            auto attn_output_ins = group;
            for(auto out : group->outputs())
            {
                auto op = out->get_operator();
                if(op.name() == "get_tuple_elem")
                {
                    inss_to_replace[out]  = out;
                    out_idx               = op.to_value()["index"].to<std::size_t>();
                    auto tuple_elem_users = out->outputs();
                    if(tuple_elem_users.size() > 1)
                        continue;

                    attn_output_ins = out;
                    out             = tuple_elem_users.front();
                }

                auto match_pw = match::match_instruction(mpm.get_module(), out, mlir_pointwise());
                if(match_pw.result != out)
                    continue;

                map_main_to_mlir_attn[attn_output_ins] = attn_outs[out_idx];
                inss_to_replace[attn_output_ins]       = out;

                auto lit_map = create_param_map_with_literals(
                    mlir_attn, out->module_inputs().front(), out->get_shape());
                mlir_attn->add_params(out->inputs(), &map_main_to_mlir_attn);
                map_main_to_mlir_attn.insert(lit_map.begin(), lit_map.end());
                std::unordered_map<instruction_ref, instruction_ref> map_pm_to_mlir_attn(
                    map_main_to_mlir_attn);
                auto fused_pw_outs = mlir_attn->fuse(
                    *out->module_inputs().front(), out->inputs(), &map_pm_to_mlir_attn);
                assert(fused_pw_outs.size() == 1);

                map_main_to_mlir_attn[out] = fused_pw_outs.front();
                attn_outs[out_idx]         = fused_pw_outs.front();
            }
        }
        mlir_attn->add_return(attn_outs);

        auto map_mlir_attn_to_main = invert_map_ins(map_main_to_mlir_attn);
        auto new_inputs            = mlir_attn->get_inputs(map_mlir_attn_to_main);

        auto mlir_ins = mpm.get_module().insert_instruction(
            group, mlir_op{make_op("dot")}, mlir_contiguous(mpm, new_inputs), {mlir_attn});

        if(inss_to_replace.empty())
        {
            mpm.get_module().replace_instruction(group, mlir_ins);
        }
        else
        {
            for(auto const& [attn_out, fused_user] : inss_to_replace)
            {
                auto replace_ins = mlir_ins;
                auto output_op   = attn_out->get_operator();
                if(output_op.name() == "get_tuple_elem")
                    replace_ins = mpm.get_module().insert_instruction(group, output_op, {mlir_ins});
                mpm.get_module().replace_instruction(fused_user, replace_ins);
            }
        }
    }
};

/**
 * Input fusion of pointwise operators into a mlir_op.
 * Only fuses unary pointwise operators by default.
 * Fuses all fusable pw ops with MIGRAPHX_ENABLE_MLIR_INPUT_FUSION
 */
struct find_pointwise_mlir
{
    auto supported_pointwise() const { return mlir_input_pointwise(match::used_once()); }

    auto matcher() const
    {
        return match::name("gpu::mlir_op")(match::any_of[match::inputs()](supported_pointwise()));
    }

    static bool is_simple_op(const_module_ref pm, std::initializer_list<std::string> op_names)
    {
        auto last = std::prev(pm->end());
        assert(last->name() == "@return");
        if(last->inputs().size() != 1)
            return false;
        auto rins   = last->inputs().front();
        auto op_ins = std::find_if(pm->begin(), pm->end(), [](const instruction& x) {
            return not contains({"@param", "@literal", "broadcast", "multibroadcast"}, x.name());
        });
        if(op_ins != rins)
            return false;
        return contains(op_names, op_ins->name());
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto* mm = ins->module_inputs().front();
        std::vector<instruction_ref> pws;
        std::copy_if(
            ins->inputs().begin(),
            ins->inputs().end(),
            std::back_inserter(pws),
            [&](instruction_ref input) {
                if(not match::instruction_matches(mpm.get_module(), input, supported_pointwise()))
                    return false;
                auto* pm = input->module_inputs().front();
                if(input->inputs().size() > 1 and not is_simple_op(pm, {"dequantizelinear"}))
                {
                    if(not enabled(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION{}))
                        return false;
                }
                return true;
            });
        if(pws.empty())
            return;

        std::string module_name;
        std::transform(
            pws.begin(), pws.end(), join_back_inserter(module_name), [](instruction_ref pw) {
                return pw->module_inputs().front()->name() + ":";
            });
        module_name += mm->name();
        module_ref m = mpm.create_module(module_name);
        m->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        for(auto pw : pws)
        {
            auto* pm = pw->module_inputs().front();
            fuse_input_ops(m, pw->inputs(), &map_ins);
            auto rins   = m->fuse(*pm, pw->inputs(), &map_ins, &insert_pointwise).front();
            map_ins[pw] = rins;
        }

        auto ret = m->fuse(*mm, ins->inputs(), &map_ins);
        m->add_return({ret});

        auto inputs = find_inputs(map_ins, &mpm.get_module(), m);
        mpm.get_module().replace_instruction(
            ins, ins->get_operator(), mlir_contiguous(mpm, inputs), {m});
    }
};

struct find_unpack_int4_mlir_op
{
    auto matcher() const
    {
        return match::name("gpu::mlir_op")(
            match::any_of[match::inputs()](match::name("unpack_int4").bind("unpack_int4")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto* mm      = ins->module_inputs().front();
        module_ref nm = mpm.create_module("int4:" + mm->name());
        nm->set_bypass();

        std::vector<instruction_ref> x_in;
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        int ct = 0;

        for(auto input : ins->inputs())
        {
            if(input->get_operator().name() == "unpack_int4")
            {
                auto unpack_input = input->inputs()[0];
                instruction_ref t_ins =
                    nm->add_parameter(param_name(++ct), unpack_input->get_shape().as_standard());
                map_ins[input] = nm->add_instruction(input->get_operator(), t_ins);
                x_in.push_back(unpack_input);
            }
            else
            {
                map_ins[input] =
                    nm->add_parameter(param_name(++ct), input->get_shape().as_standard());
                x_in.push_back(input);
            }
        }
        auto ret = nm->fuse(*mm, ins->inputs(), &map_ins);
        nm->add_return({ret});
        mpm.get_module().replace_instruction(ins, ins->get_operator(), x_in, {nm});
    }
};

/**
 * Move unpack_fp4 instructions into the mlir_op submodule.
 * Slice and reshape instructions should already be fused into the mlir_op.
 * rocMLIR will do the unpacking and dequantization.
 */
struct find_unpack_fp4_mlir_op
{
    auto matcher() const
    {
        return match::name("gpu::mlir_op")(
            match::any_of[match::inputs()](match::name("unpack_fp4")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& mr) const
    {
        auto mlir_op  = mr.result;
        auto* mm      = mlir_op->module_inputs().front();
        module_ref nm = mpm.create_module("fp4:" + mm->name());
        nm->set_bypass();
        std::vector<instruction_ref> new_mlir_op_args;
        std::unordered_map<instruction_ref, instruction_ref> fuse_ins_map;
        int ct = 0;
        for(auto curr_ins : mlir_op->inputs())
        {
            if(curr_ins->name() == "unpack_fp4")
            {
                auto unpack_ins   = curr_ins;
                auto unpack_input = unpack_ins->inputs().at(0);
                auto param =
                    nm->add_parameter(param_name(++ct), unpack_input->get_shape().as_standard());
                auto new_unpack_ins      = nm->add_instruction(unpack_ins->get_operator(), param);
                fuse_ins_map[unpack_ins] = new_unpack_ins;
                new_mlir_op_args.push_back(unpack_input);
            }
            else
            {
                fuse_ins_map[curr_ins] =
                    nm->add_parameter(param_name(++ct), curr_ins->get_shape().as_standard());
                new_mlir_op_args.push_back(curr_ins);
            }
        }
        auto ret = nm->fuse(*mm, mlir_op->inputs(), &fuse_ins_map);
        nm->add_return({ret});
        mpm.get_module().replace_instruction(
            mlir_op, mlir_op->get_operator(), new_mlir_op_args, {nm});
    }
};

/**
 * Fuse single output reshape instructions on the output of mlir_op into the
 * mlir_op.
 */
struct find_mlir_output_reshape_ops
{
    auto matcher() const
    {
        static const std::unordered_set<std::string> output_reshapes = {"transpose",
                                                                        "contiguous",
                                                                        "reshape",
                                                                        "reshape_lazy",
                                                                        "squeeze",
                                                                        "flatten",
                                                                        "unsqueeze"};
        auto atleast_one_reshape =
            match::all_of(match::output(match::name(output_reshapes)),
                          match::skip_output(match::name(output_reshapes).bind("last_reshape")));
        return match::name("gpu::mlir_op")(atleast_one_reshape);
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto mlir_op_ins     = r.result;
        auto last_reshape    = r.instructions["last_reshape"];
        auto* mlir_op_module = mlir_op_ins->module_inputs().front();
        std::vector<instruction_ref> reshape_instructions;
        instruction_ref iter_ins = mlir_op_ins;
        while(iter_ins != last_reshape)
        {
            // should already be covered by skip_output
            assert(iter_ins->outputs().size() == 1);
            auto output_ins = iter_ins->outputs().front();
            reshape_instructions.push_back(output_ins);
            iter_ins = output_ins;
        }

        assert(not reshape_instructions.empty());
        std::string module_name = mlir_op_module->name();
        std::transform(reshape_instructions.begin(),
                       reshape_instructions.end(),
                       join_back_inserter(module_name),
                       [](instruction_ref ins) { return "_" + ins->name(); });
        module_ref fused_module = mpm.create_module(module_name);
        fused_module->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        auto new_back = fused_module->fuse(*mlir_op_module, mlir_op_ins->inputs(), &map_ins).back();
        map_ins[mlir_op_ins]    = new_back;
        auto fused_instructions = fused_module->fuse(reshape_instructions, &map_ins);
        fused_module->add_return({fused_instructions.back()});
        auto inputs = find_inputs(map_ins, &mpm.get_module(), fused_module);
        mpm.get_module().replace_instruction(
            last_reshape, mlir_op{last_reshape->get_operator()}, inputs, {fused_module});
    }
};

/**
 * Find slices along the channels axis that go into a convolution.
 * Reshape input instructions to the slices such that the slice occurs over
 * the slowest dimension.
 * TODO: This can also be done for GEMM when NCHW is supported for it.
 */
struct find_channel_slice_convolution
{
    auto matcher() const
    {
        return match::name("convolution")(match::arg(0)(match::name("slice").bind("slice")));
    }

    /**
     * Number of groups the input to the slice instruction is split into.
     * `slice` should be over the channels axis only and split evenly.
     */
    static std::size_t get_num_slice_groups(instruction_ref slice)
    {
        auto input = slice->inputs().front();
        auto op    = slice->get_operator().to_value();
        auto axes  = op["axes"].to_vector<std::size_t>();
        if(axes.size() != 1)
            return 0;
        if(axes.front() != 1)
            return 0;
        auto ichannels = input->get_shape().lens().at(1);
        auto channels  = slice->get_shape().lens().at(1);
        if((ichannels % channels) != 0)
            return 0;
        return ichannels / channels;
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins              = r.result;
        auto slice            = r.instructions["slice"];
        auto input            = slice->inputs().front();
        auto num_slice_groups = get_num_slice_groups(slice);
        if(num_slice_groups == 0)
        {
            return;
        }
        // check that all slice instructions coming off from `input` are making
        // the same size slice.
        if(not all_of(input->outputs(), [&](instruction_ref output) {
               if(output->name() != "slice")
                   return false;
               auto ichannels = output->inputs().front()->get_shape().lens().at(1);
               auto channels  = output->get_shape().lens().at(1);
               return channels * num_slice_groups == ichannels;
           }))
        {
            return;
        }
        // check memory layout is in NCHW
        if(find_permutation(ins->get_shape()).back() != 1)
        {
            return;
        }

        auto dims = input->get_shape().lens();
        dims[1] /= num_slice_groups;
        // inserts num_slice_groups dimension in front of channels to split channels correctly
        dims.insert(dims.begin() + 1, num_slice_groups);

        // first transpose permutation such that channels dimension goes to
        // last axis and num_slice_groups axis goes to first axis
        std::vector<int64_t> transpose_perm0(dims.size());
        transpose_perm0.at(0) = 1;
        transpose_perm0.at(1) = 0;
        std::iota(transpose_perm0.begin() + 2, transpose_perm0.end() - 1, 3);
        transpose_perm0.back() = 2;

        // second transpose permutation such that last dimension
        // (the channels dimension below) goes to third axis
        std::vector<int64_t> transpose_perm1(dims.size());
        transpose_perm1.at(0) = 0;
        transpose_perm1.at(1) = 1;
        transpose_perm1.at(2) = dims.size() - 1;
        std::iota(transpose_perm1.begin() + 3, transpose_perm1.end(), 2);

        auto outputs       = input->outputs();
        auto ins_to_insert = std::next(input);
        auto reshape1      = mpm.get_module().insert_instruction(
            ins_to_insert, make_op("reshape_lazy", {{"dims", dims}}), input);
        auto transpose_ins1 = mpm.get_module().insert_instruction(
            ins_to_insert, make_op("transpose", {{"permutation", transpose_perm0}}), reshape1);
        auto contiguous_ins = mpm.get_module().insert_instruction(
            ins_to_insert, make_op("contiguous"), transpose_ins1);
        auto transepose_ins2 = mpm.get_module().insert_instruction(
            ins_to_insert,
            make_op("transpose", {{"permutation", transpose_perm1}}),
            contiguous_ins);
        // spacer identity instruction
        auto identity_ins = mpm.get_module().insert_instruction(
            ins_to_insert, make_op("identity"), transepose_ins2);

        // Replace slice operators to 0 axis
        for(auto output : outputs)
        {
            auto v      = output->get_operator().to_value();
            auto starts = v["starts"].to_vector<std::size_t>();
            auto i      = starts.front() / output->get_shape().lens()[1]; // note integer truncation
            auto s      = mpm.get_module().insert_instruction(
                output,
                make_op("slice", {{"axes", {0}}, {"starts", {i}}, {"ends", {i + 1}}}),
                identity_ins);
            mpm.get_module().replace_instruction(output, make_op("squeeze", {{"axes", {0}}}), s);
        }
    }
};

} // namespace

#endif // MIGRAPHX_MLIR

void fuse_mlir::apply(module_pass_manager& mpm) const
{
#ifdef MIGRAPHX_MLIR
    std::size_t counter     = 0;
    const auto& device_name = ctx == nullptr ? "" : ctx->get_current_device().get_gfx_name();
    const bool is_navi = starts_with(device_name, "gfx11") or starts_with(device_name, "gfx12");

    auto get_mode = [&](std::string_view option, mlir_mode m1, mlir_mode m2 = mlir_mode::fast) {
        if(specific_op<rejected>(option))
            return mlir_mode::none;
        if(specific_op<requested>(option))
            return mlir_mode::all;
        if(is_navi)
            return mlir_mode::all;
        return std::max(m1, m2);
    };

    match::find_matches(mpm, find_channel_slice_convolution{});
    mpm.run_pass(dead_code_elimination{});

    match::find_matches(mpm, find_mlir_kv_cache_attention_op{mlir_mode::all});
    mpm.run_pass(dead_code_elimination{});

    match::find_matches(mpm, find_mlir_attention_op{});
    mpm.run_pass(dead_code_elimination{});

    match::find_matches(
        mpm,
        find_mlir_fused_ops{.conv_mode = get_mode("fused_convolution", mlir_mode::fast),
                            .dot_mode  = get_mode("fused_dot", mlir_mode::fast)});

    match::find_matches(
        mpm,
        find_mlir_standalone_conv_op{.mode    = get_mode("convolution", mlir_mode::fast),
                                     .counter = &counter},
        find_mlir_standalone_conv_backwards_op{
            .mode    = get_mode("convolution_backwards",
                             MIGRAPHX_USE_MIOPEN ? mlir_mode::none : mlir_mode::all),
            .counter = &counter},
        find_mlir_standalone_dot_op{.mode = get_mode("dot", mlir_mode::fast), .counter = &counter});

    mpm.run_pass(dead_code_elimination{});

    // GEG fusion runs after standalone gemm and gemm->pointwise fusions, since it combines them
    mlir_mode geg_dot_mode =
        enabled(MIGRAPHX_DISABLE_MLIR_GEG_FUSION{}) ? mlir_mode::none : mlir_mode::fast;
    mlir_mode geg_conv_mode =
        enabled(MIGRAPHX_ENABLE_MLIR_CEG_FUSION{}) ? mlir_mode::fast : mlir_mode::none;

    match::find_matches(mpm,
                        find_mlir_fused_geg_ops{.conv_mode = geg_conv_mode,
                                                .dot_mode  = geg_dot_mode,
                                                .gfx_name  = device_name});
    mpm.run_pass(dead_code_elimination{});

    if(enabled(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION{}))
    {
        match::find_matches(
            mpm,
            find_mlir_split_reduce{.conv_mode = get_mode("fused_convolution", mlir_mode::fast),
                                   .dot_mode  = get_mode("fused_dot", mlir_mode::fast)});
    }

    match::find_matches(mpm, find_pointwise_mlir{});
    match::find_matches(mpm, find_unpack_int4_mlir_op{});
    match::find_matches(mpm, find_unpack_fp4_mlir_op{});

    match::find_matches(mpm, find_mlir_output_reshape_ops{});

#else
    (void)mpm;
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
