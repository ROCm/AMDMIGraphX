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
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_ops.hpp>
#include <migraphx/gpu/precompile_op.hpp>
#include <migraphx/gpu/compile_hipblaslt.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/hip_gemm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/array.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <bitset>
#include <cmath>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_MIOPEN_FUSION)
#if MIGRAPHX_USE_MIOPEN
struct fusion
{
    using op_t = miopenFusionOpDescriptor_t;
    shared<fusion_plan_descriptor> fp;

    // Used as a temporary hack to keep descriptor references alive
    std::vector<std::shared_ptr<void>> storage;

    template <class T>
    auto keep_alive(T x)
    {
        auto result = share(std::move(x));
        storage.push_back(result);
        return result;
    }

    fusion() = default;

    fusion(const shape& input)
    {
        assert(input.standard());
        auto t = make_tensor(input);
        fp     = make_fusion_plan(t);
        assert(fp);
        keep_alive(std::move(t));
    }

    bool empty() const { return fp == nullptr; }

    op_t operator[](std::size_t i) const
    {
        assert(fp);
        op_t result;
        auto status = miopenFusionPlanGetOp(fp.get(), i, &result);
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Failed retrieving operator at " + std::to_string(i));
        return result;
    }

    auto get() const
    {
        assert(fp);
        return fp.get();
    }

    op_t create_bias(const shape& bias)
    {
        assert(fp);
        op_t result;
        auto b      = shape{bias.type(), {1, bias.lens().at(1), 1, 1}};
        auto t      = keep_alive(make_tensor(b));
        auto status = miopenCreateOpBiasForward(fp.get(), &result, t.get());
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Creating operator failed");
        return result;
    }

    op_t create_relu()
    {
        assert(fp);
        op_t result;
        auto status = miopenCreateOpActivationForward(fp.get(), &result, miopenActivationRELU);
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Creating operator failed");
        return result;
    }

    op_t create_conv(const op::convolution& op, const shape& weights)
    {
        assert(fp);
        op_t result;
        auto cd     = keep_alive(make_conv(op));
        auto t      = keep_alive(make_tensor(weights));
        auto status = miopenCreateOpConvForward(fp.get(), &result, cd.get(), t.get());
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Creating operator failed");
        return result;
    }

    shape get_workspace(context&)
    {
        // assert(fp);
        // TODO: Use zero workspace for now
        std::size_t ws_size = 0;
        // int algo_count = 1;
        // miopenConvFwdAlgorithm_t algo;
        // miopenFusionPlanConvolutionGetAlgo(fp.get(), 1, &algo_count, &algo);
        // miopenFusionPlanGetWorkSpaceSize(ctx.get_stream().get_miopen(), fp.get(), &ws_size,
        // algo);
        return shape{shape::int8_type, {ws_size}};
    }

    bool compile(context& ctx)
    {
        assert(fp);
        return miopenCompileFusionPlan(ctx.get_stream().get_miopen(), fp.get()) ==
               miopenStatusSuccess;
    }

    argument execute(context& ctx,
                     const fused_operator_args& fargs,
                     const argument& x,
                     const argument& y) const
    {
        assert(fp);
        auto x_td   = make_tensor(x.get_shape());
        auto y_td   = make_tensor(y.get_shape());
        auto status = miopenExecuteFusionPlan(ctx.get_stream().get_miopen(),
                                              fp.get(),
                                              x_td.get(),
                                              x.implicit(),
                                              y_td.get(),
                                              y.implicit(),
                                              fargs.get());
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Failed to execute fusion plan");
        return y;
    }
};
#endif

#if MIGRAPHX_USE_MIOPEN
static const std::unordered_set<std::string>& get_supported_archs()
{
    static std::unordered_set<std::string> supported_archs{"gfx900", "gfx906", "gfx908", "gfx1030"};
    return supported_archs;
}
MIGRAPHX_PRED_MATCHER(bias_shape, instruction_ref ins)
{
    auto&& s = ins->get_shape();
    return s.broadcasted() and s.strides().size() == 4 and s.strides()[0] == 0 and
           s.strides()[1] != 0 and s.strides()[2] == 0 and s.strides()[3] == 0;
}

MIGRAPHX_PRED_MATCHER(fusable_conv, instruction_ref ins)
{
    const auto device_name = trim(split_string(get_device_name(), ':').front());
    if(not contains(get_supported_archs(), device_name))
        return false;
    if(enabled(MIGRAPHX_DISABLE_MIOPEN_FUSION{}))
        return false;
    if(ins->name() != "gpu::convolution")
        return false;
    if(ins->get_shape().type() != shape::float_type)
        return false;
    auto wei = ins->inputs().at(1)->get_shape();
    assert(wei.lens().size() == 4);
    auto miopen_conv_op = ins->get_operator().to_value();
    auto algo           = miopen_conv_op.at("algo").to<miopenConvFwdAlgorithm_t>();
    auto conv_op        = from_value<op::convolution>(miopen_conv_op["op"]);
    if(conv_op.group > 1)
        return false;
    if(wei.lens()[1] > 512 and algo != miopenConvolutionFwdAlgoWinograd)
        return false;

    // Do not fuse non-symmetric input
    auto input_lens = ins->inputs().at(0)->get_shape().lens();
    if(input_lens[2] != input_lens[3] or wei.lens()[2] != wei.lens()[3])
        return false;

    // Dont fuse winograd for non-3x3s since there is no fused windograd for those configs
    if(algo == miopenConvolutionFwdAlgoWinograd and wei.lens()[2] != 3 and wei.lens()[3] != 3 and
       contains({{1, 1}}, conv_op.stride))
        return false;
    return contains({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}}, conv_op.padding) and
           contains({{0, 0}, {1, 1}}, conv_op.stride) and contains({{1, 1}}, conv_op.dilation);
}
#endif

static void move_broadcasted_back(std::vector<instruction_ref>& args)
{
    // Ensure the last arguments is the broadcasted one
    auto last = std::prev(args.end());
    auto it =
        std::find_if(args.begin(), last, [](auto arg) { return arg->get_shape().broadcasted(); });
    if(it != last)
        std::swap(*it, *std::prev(last));
}

namespace {
#if MIGRAPHX_USE_MIOPEN
struct miopen_fusion
{
    struct fuse_op_data
    {
        operation op;
        float alpha = 1;
        float beta  = 0;
    };
    struct fuse_op : fuse_op_data, reflect_equality<fuse_op>, reflect_stream<fuse_op>
    {
        template <class Self, class F>
        static auto reflect(Self& self, F f)
        {
            return pack(f(self.op, "op"), f(self.alpha, "alpha"), f(self.beta, "beta"));
        }
    };
    std::vector<fuse_op> ops = {};
    fusion f                 = {};
    std::function<void(context&, const fusion&, const std::vector<argument>&)> execute;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.ops, "ops"));
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        return {shapes.size() - 1};
    }

    value compile(context& ctx, const shape&, std::vector<shape> inputs)
    {
        // Compensate for allocation
        inputs.pop_back();
        std::size_t i = 0;
        f             = fusion(inputs[i]);
        i++;
        std::vector<std::function<void(const fused_operator_args&, const std::vector<argument>&)>>
            invokers;
        for(auto&& fop : ops)
        {
            if(i > inputs.size())
            {
                f = {};
                return {};
            }
            if(fop.op.name() == "convolution")
            {
                auto* mop = f.create_conv(any_cast<op::convolution>(fop.op), inputs[i]);
                invokers.push_back(
                    [=](const fused_operator_args& fargs, const std::vector<argument>& args) {
                        miopenSetOpArgsConvForward(
                            fargs.get(), mop, &fop.alpha, &fop.beta, args[i].implicit());
                    });
                i++;
            }
            else if(fop.op.name() == "add")
            {
                auto* mop = f.create_bias(inputs[i]);
                invokers.push_back(
                    [=](const fused_operator_args& fargs, const std::vector<argument>& args) {
                        miopenSetOpArgsBiasForward(
                            fargs.get(), mop, &fop.alpha, &fop.beta, args[i].implicit());
                    });
                i++;
            }
            else if(fop.op.name() == "relu")
            {
                auto* mop = f.create_relu();
                invokers.push_back([=](const fused_operator_args& fargs,
                                       const std::vector<argument>&) {
                    miopenSetOpArgsActivForward(fargs.get(), mop, &fop.alpha, &fop.beta, 0, 0, 0);
                });
            }
            else
            {
                f = {};
                return {};
            }
        }
        if(not f.compile(ctx))
        {
            f = {};
            return {};
        }
        execute = [invokers](context& c, const fusion& ff, const std::vector<argument>& args) {
            auto fargs = make_fused_args();
            for(auto&& invoker : invokers)
                invoker(fargs, args);
            ff.execute(c, fargs, args.front(), args.back());
        };
        return {{"workspace", f.get_workspace(ctx).bytes()}};
    }
    void finalize(context& ctx, const shape& output_shape, const std::vector<shape>& inputs)
    {
        if(not f.empty())
            return;
        auto v = compile(ctx, output_shape, inputs);
        if(not v.is_object())
            MIGRAPHX_THROW("Failed to compile fusion plan");
    }
    std::string name() const { return "gpu::miopen_fusion"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        if(ops.empty())
            return {};
        // TODO: Check number of arguments
        return ops.front().op.compute_shape({inputs[0], inputs[1]});
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        execute(ctx, f, args);
        return args.back();
    }
};
MIGRAPHX_REGISTER_OP(miopen_fusion)

struct miopen_conv_bias
{
    op::convolution op;
    fusion fp         = {};
    fusion::op_t conv = {};
    fusion::op_t bias = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return op::convolution::reflect(self.op, f);
    }

    std::string name() const { return "gpu::conv_bias"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        // TODO: Check slices
        return op.normalize_compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        auto fargs  = make_fused_args();
        float alpha = 1;
        float beta  = 0;
        miopenSetOpArgsConvForward(fargs.get(), conv, &alpha, &beta, args[1].implicit());
        miopenSetOpArgsBiasForward(fargs.get(), bias, &alpha, &beta, args[3].implicit());
        return fp.execute(ctx, fargs, args[0], args[4]);
    }

    void finalize(context& ctx, const shape&, const std::vector<shape>& inputs)
    {
        fp   = fusion(inputs[0]);
        conv = fp.create_conv(op, inputs[1]);
        bias = fp.create_bias(inputs[3]);
        if(not fp.compile(ctx))
            MIGRAPHX_THROW("Failed to compile fusion plan");
    }

    shape get_workspace(context& ctx) { return fp.get_workspace(ctx); }
    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        return {shapes.size() - 1};
    }
};
MIGRAPHX_REGISTER_OP(miopen_conv_bias)

struct miopen_conv_bias_relu
{
    op::convolution op;
    fusion fp         = {};
    fusion::op_t conv = {};
    fusion::op_t bias = {};
    fusion::op_t relu = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return op::convolution::reflect(self.op, f);
    }

    std::string name() const { return "gpu::conv_bias_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        // TODO: Check slices
        return op.normalize_compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        auto fargs  = make_fused_args();
        float alpha = 1;
        float beta  = 0;
        miopenSetOpArgsConvForward(fargs.get(), conv, &alpha, &beta, args[1].implicit());
        miopenSetOpArgsBiasForward(fargs.get(), bias, &alpha, &beta, args[3].implicit());
        miopenSetOpArgsActivForward(fargs.get(), relu, &alpha, &beta, 0, 0, 0);
        return fp.execute(ctx, fargs, args[0], args[4]);
    }
    void finalize(context& ctx, const shape&, const std::vector<shape>& inputs)
    {
        fp   = fusion(inputs[0]);
        conv = fp.create_conv(op, inputs[1]);
        bias = fp.create_bias(inputs[3]);
        relu = fp.create_relu();
        fp.compile(ctx);
    }

    shape get_workspace(context& ctx) { return fp.get_workspace(ctx); }
    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        return {shapes.size() - 1};
    }
};
MIGRAPHX_REGISTER_OP(miopen_conv_bias_relu)

template <class... Ms>
auto conv_bias(Ms... ms)
{
    return match::name("gpu::add")(
        match::either_arg(0, 1)(bias_shape(match::used_once()).bind("bias"),
                                fusable_conv(match::used_once()).bind("conv")),
        ms...);
}

template <class Op>
void apply_conv_bias(context& ctx, module& m, const match::matcher_result& r)
{
    auto conv_ins    = r.instructions["conv"];
    auto bias_ins    = r.instructions["bias"];
    auto ins         = r.result;
    auto input_ins   = conv_ins->inputs().at(0);
    auto weights_ins = conv_ins->inputs().at(1);
    auto conv_op     = from_value<op::convolution>((conv_ins->get_operator()).to_value()["op"]);
    auto alloc_ins   = ins->inputs().back();
    auto old_ws_ins  = conv_ins->inputs().at(2);

    Op cb{conv_op};
    // TODO: Insert ws allocation
    auto ws = cb.get_workspace(ctx);
    (void)ws;
    m.replace_instruction(ins, cb, input_ins, weights_ins, old_ws_ins, bias_ins, alloc_ins);
}
#endif

#if MIGRAPHX_USE_MIOPEN
struct find_conv_bias
{
    context* ctx = nullptr;
    auto matcher() const
    {
        auto relu = match::name(std::unordered_set<std::string>{"gpu::relu"});
        return conv_bias(match::none_of(match::output(relu)));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        apply_conv_bias<miopen_conv_bias>(*ctx, m, r);
    }
};

struct find_conv_bias_relu
{
    context* ctx = nullptr;
    auto matcher() const { return match::name("gpu::relu")(match::arg(0)(conv_bias())); }

    void apply(module& m, const match::matcher_result& r) const
    {
        apply_conv_bias<miopen_conv_bias_relu>(*ctx, m, r);
    }
};
struct find_conv_pointwise
{
    context* ctx = nullptr;
    auto matcher() const
    {
        return precompile_name("pointwise")(
            match::nargs(3),
            match::either_arg(0, 1)(bias_shape(match::used_once()).bind("bias"),
                                    fusable_conv(match::used_once()).bind("conv")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto conv_ins    = r.instructions["conv"];
        auto bias_ins    = r.instructions["bias"];
        auto ins         = r.result;
        auto input_ins   = conv_ins->inputs().at(0);
        auto weights_ins = conv_ins->inputs().at(1);
        auto conv_op     = from_value<op::convolution>(conv_ins->get_operator().to_value()["op"]);
        auto alloc_ins   = ins->inputs().back();

        module_ref pm = ins->module_inputs().front();

        miopen_fusion op{};
        op.ops.push_back({{conv_op}});
        for(auto&& i : *pm)
        {
            if(i.name()[0] == '@')
                continue;
            op.ops.push_back({{i.get_operator()}});
        }
        std::vector<instruction_ref> inputs = {input_ins, weights_ins, bias_ins, alloc_ins};
        auto v                              = op.compile(*ctx, ins->get_shape(), to_shapes(inputs));
        if(not v.is_object())
            return;
        m.replace_instruction(ins, op, inputs);
    }
};
#endif

#if MIGRAPHX_USE_ROCBLAS or MIGRAPHX_USE_HIPBLASLT
struct gemm_pointwise
{
    // TODO: Move to matcher.hpp
    static auto match_param(const std::string& name)
    {
        return match::make_basic_pred_matcher([=](auto ins) {
            if(ins->name() != "@param")
                return false;
            auto p = any_cast<builtin::param>(ins->get_operator());
            return p.parameter == name;
        });
    }

    template <class M>
    static auto match_mul_const(M m, const std::string& var)
    {
        return match::name("mul")(match::either_arg(0, 1)(match::name("@literal").bind(var), m))
            .bind(var + "_mul");
    }

    static auto match_add(const std::string& input, const std::string& output)
    {
        auto param     = match::name("@param");
        auto add       = match::name("add")(match::args(param, param));
        auto inner_mul = match::any_of(match_mul_const(match_param(input), "alpha"),
                                       match_mul_const(match_param(output), "beta"));
        auto mul_add   = match::name("add")(match::either_arg(0, 1)(inner_mul, param));
        auto add_mul   = match_mul_const(add, "gamma");
        return match::name("@return")(match::args(match::any_of(add, mul_add, add_mul)));
    }

    static auto match_mul(const std::string& input)
    {
        auto mul = match_mul_const(match_param(input), "alpha");
        return match::name("@return")(match::args(mul));
    }

    static float get_float(instruction_ref ins) { return ins->get_literal().at<float>(); }

    template <class Gemm>
    static bool update_gemm(Gemm& gemm, module_ref pm, unsigned input)
    {
        auto names = pm->get_parameter_names();
        std::sort(names.begin(), names.end());
        if(names.size() == 1)
        {
            auto mr = match::match_instruction(*pm, std::prev(pm->end()), match_mul(names[input]));
            if(mr.result == pm->end())
                return false;
            gemm.alpha *= get_float(mr.instructions["alpha"]);
            return true;
        }
        else if(names.size() == 2)
        {
            unsigned output = input == 0 ? 1 : 0;
            auto mr         = match::match_instruction(
                *pm, std::prev(pm->end()), match_add(names[input], names[output]));
            if(mr.result == pm->end())
                return false;
            if(contains(mr.instructions, "alpha_mul"))
                gemm.alpha *= get_float(mr.instructions["alpha"]);
            else if(contains(mr.instructions, "beta_mul"))
                gemm.beta *= get_float(mr.instructions["beta"]);
            else if(contains(mr.instructions, "gamma_mul"))
            {
                gemm.alpha *= get_float(mr.instructions["gamma"]);
                gemm.beta *= get_float(mr.instructions["gamma"]);
            }
            return true;
        }
        else
        {
            return false;
        }
    }
};
#endif

#if MIGRAPHX_USE_ROCBLAS
struct find_rocblas_gemm_pointwise : gemm_pointwise
{
    auto matcher() const
    {
        auto gemm_op   = match::name("gpu::gemm")(match::nargs(3), match::used_once()).bind("gemm");
        auto binary_op = match::all_of(
            match::nargs(3),
            match::either_arg(0, 1)(
                match::any_of(match::standard_shape(), match::is_constant()).bind("c"), gemm_op));
        auto unary_op = match::all_of(match::nargs(2), match::arg(0)(gemm_op));
        return precompile_name("pointwise")(match::any_of(binary_op, unary_op));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto gemm_ins = r.instructions["gemm"];

        auto gemm = any_cast<rocblas_gemm<op::dot>>(gemm_ins->get_operator());

        // Already fused gemm
        if(not float_equal(gemm.beta, 0))
            return;
        if(ins->inputs().size() == 3)
            gemm.beta = 1;

        if(not update_gemm(
               gemm, ins->module_inputs().front(), ins->inputs().front() == gemm_ins ? 0 : 1))
            return;

        auto inputs = gemm_ins->inputs();
        inputs.pop_back();

        if(ins->inputs().size() == 3)
        {
            auto c_ins = r.instructions["c"];
            shape s    = c_ins->get_shape();
            // const-fold input if not standard shape since rocblas can't handle it
            if(not s.standard())
            {
                auto c = make_op("contiguous");
                auto l = c.compute(c.compute_shape({c_ins->get_shape()}), {c_ins->eval()});
                c_ins  = m.add_literal(literal::from_raw_buffer(l.get_shape(), l.data()));
            }
            inputs.push_back(c_ins);
        }

        inputs.push_back(ins->inputs().back());

        m.replace_instruction(ins, gemm, inputs);
    }
};
#endif

#if MIGRAPHX_USE_HIPBLASLT
struct find_hipblas_gemm_pointwise : gemm_pointwise
{
    auto matcher() const
    {
        auto gemm_op =
            match::name("gpu::hipblaslt_op")(match::nargs(3), match::used_once()).bind("hip_gemm");
        auto binary_op = match::all_of(
            match::nargs(3),
            match::either_arg(0, 1)(
                match::any_of(match::standard_shape(), match::is_constant()).bind("c"), gemm_op));
        auto unary_op = match::all_of(match::nargs(2), match::arg(0)(gemm_op));
        return precompile_name("pointwise")(match::any_of(binary_op, unary_op));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto gemm_ins = r.instructions["hip_gemm"];

        auto gemm_op = any_cast<hipblaslt_op>(gemm_ins->get_operator()).op;

        if(gemm_op.name() != "gpu::hip_gemm")
            return;

        auto gemm = any_cast<hip_gemm<op::dot>>(gemm_op);

        // Already fused gemm
        if(not float_equal(gemm.beta, 0))
            return;
        if(ins->inputs().size() == 3)
            gemm.beta = 1;
        if(not update_gemm(
               gemm, ins->module_inputs().front(), ins->inputs().front() == gemm_ins ? 0 : 1))
        {
            return;
        }
        auto inputs = gemm_ins->inputs();
        inputs.pop_back();
        if(ins->inputs().size() == 3)
        {
            auto c_ins = r.instructions["c"];
            shape s    = c_ins->get_shape();
            // const-fold input if not standard shape
            if(not s.standard())
            {
                auto c = make_op("contiguous");
                auto l = c.compute(c.compute_shape({c_ins->get_shape()}), {c_ins->eval()});
                c_ins  = m.add_literal(literal::from_raw_buffer(l.get_shape(), l.data()));
            }
            inputs.push_back(c_ins);
        }
        inputs.push_back(ins->inputs().back());

        operation new_gemm_op = gemm;
        auto new_ins          = m.insert_instruction(
            ins, make_op("gpu::hipblaslt_op", {{"op", to_value(new_gemm_op)}}), inputs);
        m.replace_instruction(ins, new_ins);
    }
};
#endif

struct contiguous_transpose_gemm
{
    template <class Vector>
    static bool is_swapped(const Vector& perm, std::size_t i, std::size_t j)
    {
        if(i >= perm.size() or j >= perm.size())
            return false;
        auto perm2 = perm;
        std::iota(perm2.begin(), perm2.end(), 0);
        std::swap(perm2[i], perm2[j]);
        return perm2 == perm;
    }
};

struct find_contiguous_transpose_rocblas_gemm : contiguous_transpose_gemm
{
    auto matcher() const
    {
        return match::name("gpu::contiguous")(match::arg(0)(
            match::name("transpose")(
                match::arg(0)(match::name("gpu::gemm")(match::used_once()).bind("gemm")))
                .bind("transpose")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins       = r.result;
        auto gemm      = r.instructions["gemm"];
        auto alloc     = gemm->inputs().back();
        auto transpose = r.instructions["transpose"];
        auto perm      = transpose->get_operator().to_value()["permutation"].to_vector<int64_t>();
        auto iperm     = invert_permutation(perm);

        if(perm.size() < 3)
            return;

        if(not is_swapped(perm, perm.size() - 3, perm.size() - 2))
            return;

        auto lens = gemm->get_shape().lens();
        if(lens.size() > 3 and
           not std::all_of(lens.begin(), lens.end() - 3, [](auto i) { return i == 1; }))
            return;

        auto gemmv           = gemm->get_operator().to_value();
        gemmv["trans_batch"] = 1;

        auto s = shape{alloc->get_shape().type(), reorder_dims(alloc->get_shape().lens(), iperm)};
        auto new_alloc = m.insert_instruction(gemm, make_op("allocate", {{"shape", to_value(s)}}));
        auto alloc_transpose =
            m.insert_instruction(gemm, make_op("transpose", {{"permutation", perm}}), new_alloc);

        auto inputs        = gemm->inputs();
        inputs.back()      = alloc_transpose;
        auto new_gemm      = m.insert_instruction(gemm, make_op("gpu::gemm", gemmv), inputs);
        auto gemm_transpoe = m.insert_instruction(gemm, transpose->get_operator(), new_gemm);

        m.replace_instruction(ins, gemm_transpoe);
    }
};

#if MIGRAPHX_USE_HIPBLASLT
struct find_contiguous_transpose_hip_gemm : contiguous_transpose_gemm
{
    auto matcher() const
    {
        return match::name("gpu::contiguous")(match::arg(0)(
            match::name("transpose")(
                match::arg(0)(
                    match::name("gpu::hipblaslt_op")(match::used_once()).bind("hip_gemm")))
                .bind("transpose")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto gemm_ins = r.instructions["hip_gemm"];
        auto gemm_op  = any_cast<hipblaslt_op>(gemm_ins->get_operator()).op;

        if(gemm_op.name() != "gpu::hip_gemm")
            return;

        auto gemm = any_cast<hip_gemm<op::dot>>(gemm_op);

        auto alloc     = gemm_ins->inputs().back();
        auto transpose = r.instructions["transpose"];
        auto perm      = transpose->get_operator().to_value()["permutation"].to_vector<int64_t>();
        auto iperm     = invert_permutation(perm);

        if(perm.size() < 3)
            return;

        if(not is_swapped(perm, perm.size() - 3, perm.size() - 2))
            return;

        auto lens = gemm_ins->get_shape().lens();
        if(lens.size() > 3 and
           not std::all_of(lens.begin(), lens.end() - 3, [](auto i) { return i == 1; }))
            return;

        gemm.trans_batch = 1;

        auto s = shape{alloc->get_shape().type(), reorder_dims(alloc->get_shape().lens(), iperm)};
        auto new_alloc =
            m.insert_instruction(gemm_ins, make_op("allocate", {{"shape", to_value(s)}}));

        auto alloc_transpose = m.insert_instruction(
            gemm_ins, make_op("transpose", {{"permutation", perm}}), new_alloc);

        auto inputs           = gemm_ins->inputs();
        inputs.back()         = alloc_transpose;
        operation new_gemm_op = gemm;
        auto new_gemm         = m.insert_instruction(
            gemm_ins, make_op("gpu::hipblaslt_op", {{"op", to_value(new_gemm_op)}}), inputs);

        auto gemm_transpoe = m.insert_instruction(gemm_ins, transpose->get_operator(), new_gemm);

        m.replace_instruction(ins, gemm_transpoe);
    }
};
#endif

struct find_commutative_broadcast
{
    auto matcher() const
    {
        return match::name("gpu::add", "gpu::mul")(match::arg(1)(match::broadcast_shape()));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins  = r.result;
        auto args = ins->inputs();
        move_broadcasted_back(args);

        m.replace_instruction(ins, ins->get_operator(), args);
    }
};
} // namespace

struct find_contiguous_layout_pointwise
{
    auto matcher() const
    {
        auto cont_pw   = precompile_name("pointwise")(match::any_of[match::inputs()](
            match::name("gpu::contiguous")(match::used_once()).bind("layout_ins")));
        auto layout_pw = precompile_name("pointwise")(match::any_of[match::inputs()](
            precompile_name("layout")(match::used_once()).bind("layout_ins")));
        return match::any_of(cont_pw, layout_pw);
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto pw_ins        = r.result;
        auto layout_ins    = r.instructions["layout_ins"];
        auto layout_input  = layout_ins->inputs().front();
        auto pw_ins_inputs = pw_ins->inputs();
        replace(pw_ins_inputs, layout_ins, layout_input);
        // Ensure the output shape of the pointwise module retains the memory layout
        auto pw_op_val            = pw_ins->get_operator().to_value();
        pw_op_val["output_shape"] = to_value(pw_ins->get_shape());

        auto new_ins = m.insert_instruction(
            pw_ins, make_op(pw_ins->name(), pw_op_val), pw_ins_inputs, pw_ins->module_inputs());
        m.replace_instruction(pw_ins, new_ins);
    }
};

struct find_pointwise_layout_contiguous
{
    auto matcher() const
    {
        auto is_layout = precompile_name("layout")(
            match::arg(0)(match::used_once(), precompile_name("pointwise")));
        auto is_contiguous = match::name("gpu::contiguous")(
            match::arg(0)(match::used_once(), precompile_name("pointwise")));
        return match::any_of(is_layout, is_contiguous);
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto pw     = ins->inputs().front();
        auto alloc  = ins->inputs().back();
        auto args   = pw->inputs();
        args.back() = alloc;

        // Ensure the output shape of the pointwise module retains the memory layout
        auto pw_op_val            = pw->get_operator().to_value();
        pw_op_val["output_shape"] = to_value(ins->get_shape());

        m.replace_instruction(ins, make_op(pw->name(), pw_op_val), args, pw->module_inputs());
    }
};

struct find_precompile_copy
{
    auto matcher() const
    {
        return match::name("hip::copy")(
            match::arg(0)(match::used_once(), match::name("gpu::precompile_op")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto pw    = ins->inputs().front();
        auto alloc = ins->inputs().back();

        // Only merge when the precompile_op's current output buffer is a plain
        // allocation.  After eliminate_concat the buffer may be a slice of a
        // shared super-allocation; merging a second copy on top would redirect
        // the write and leave the first destination unwritten.
        auto current_buf = pw->inputs().back();
        if(current_buf->name() != "allocate" and current_buf->name() != "hip::allocate")
            return;

        auto args   = pw->inputs();
        args.back() = alloc;

        // Ensure the output shape of the precompile op retains the memory layout
        auto pw_op_val            = pw->get_operator().to_value();
        pw_op_val["output_shape"] = to_value(ins->get_shape());

        m.replace_instruction(ins, make_op(pw->name(), pw_op_val), args, pw->module_inputs());
    }
};

struct find_layernorm_pointwise
{
    auto matcher() const
    {
        return precompile_name("pointwise")(
            match::not_tuple(),
            match::arg(0)(
                precompile_name("gpu::prelayernorm", "gpu::preadd_layernorm").bind("layernorm")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto pw_ins    = r.result;
        auto layernorm = r.instructions["layernorm"];
        if(not layernorm->module_inputs().empty())
            return;
        auto* pm       = pw_ins->module_inputs().front();
        auto pw_inputs = pw_ins->inputs();
        auto ln_pos    = std::find(pw_inputs.begin(), pw_inputs.end(), layernorm);
        assert(ln_pos != pw_inputs.end());
        pw_inputs.erase(ln_pos);
        auto inputs = layernorm->inputs();
        inputs.pop_back();
        inputs.insert(inputs.end(), pw_inputs.begin(), pw_inputs.end());

        // Ensure the output shape retains the memory layout
        auto layernorm_op_val            = layernorm->get_operator().to_value();
        layernorm_op_val["output_shape"] = to_value(pw_ins->get_shape());

        m.replace_instruction(pw_ins, make_op(layernorm->name(), layernorm_op_val), inputs, {pm});
    }
};

struct find_channelwise_conv_pointwise
{
    auto matcher() const
    {
        return precompile_name("pointwise")(
            match::not_tuple(),
            match::arg(0)(precompile_name("gpu::channelwise_conv").bind("channelwise_conv")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto pw_ins          = r.result;
        auto channelwise_ins = r.instructions["channelwise_conv"];
        if(not channelwise_ins->module_inputs().empty())
            return;
        auto* pm       = pw_ins->module_inputs().front();
        auto pw_inputs = pw_ins->inputs();
        auto cw_pos    = std::find(pw_inputs.begin(), pw_inputs.end(), channelwise_ins);
        assert(cw_pos != pw_inputs.end());
        pw_inputs.erase(cw_pos);
        auto inputs = channelwise_ins->inputs();
        inputs.pop_back();
        inputs.insert(inputs.end(), pw_inputs.begin(), pw_inputs.end());

        auto cw_op_val            = channelwise_ins->get_operator().to_value();
        cw_op_val["output_shape"] = to_value(pw_ins->get_shape());

        m.replace_instruction(pw_ins, make_op(channelwise_ins->name(), cw_op_val), inputs, {pm});
    }
};

// Fuse a pointwise op that consumes a winograd_conv output. Replaces
// `pointwise(winograd_conv(...), extras...)` with a single winograd_conv
// instance whose precompile_op has the pointwise module attached. The kernel
// then applies the pointwise function inline during writeback at no extra
// memory traffic.
struct find_winograd_conv_pointwise
{
    auto matcher() const
    {
        return precompile_name("pointwise")(
            match::not_tuple(),
            match::arg(0)(precompile_name("gpu::winograd_conv").bind("winograd_conv")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto pw_ins       = r.result;
        auto winograd_ins = r.instructions["winograd_conv"];
        if(not winograd_ins->module_inputs().empty())
            return;
        assert(not pw_ins->module_inputs().empty());
        auto* pm       = pw_ins->module_inputs().front();
        auto pw_inputs = pw_ins->inputs();
        auto wn_pos    = std::find(pw_inputs.begin(), pw_inputs.end(), winograd_ins);
        assert(wn_pos != pw_inputs.end());
        pw_inputs.erase(wn_pos);
        // Winograd's inputs are (x, u, output_alloc). Drop that alloc, then
        // append the remaining pointwise inputs (bias, ..., pointwise's own
        // output_alloc) — that alloc stays last, where precompile_op expects it.
        auto inputs = winograd_ins->inputs();
        assert(not inputs.empty());
        inputs.pop_back();
        inputs.insert(inputs.end(), pw_inputs.begin(), pw_inputs.end());

        auto wn_op_val            = winograd_ins->get_operator().to_value();
        wn_op_val["output_shape"] = to_value(pw_ins->get_shape());

        m.replace_instruction(pw_ins, make_op(winograd_ins->name(), wn_op_val), inputs, {pm});
    }
};

struct find_concat_pointwise
{
    auto matcher() const
    {
        return precompile_name("pointwise")(
            match::arg(0)(precompile_name("concat").bind("concat"), match::used_once()));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto concat = r.instructions["concat"];
        if(not concat->module_inputs().empty())
            return;

        // TODO: Handle type conversions
        if(ins->get_shape().type() != concat->get_shape().type())
            return;

        auto* pm    = ins->module_inputs().front();
        auto inputs = concat->inputs();
        inputs.pop_back();
        inputs.insert(inputs.end(), ins->inputs().begin() + 1, ins->inputs().end());

        auto op = concat->get_operator();
        op.from_value({{"additional_args", ins->inputs().size() - 1},
                       {"ignore_modules", true},
                       {"output_shape", to_value(ins->get_shape())}});

        m.replace_instruction(ins, op, inputs, {pm});
    }
};

// The attributes of the operator a precompile_op wraps
static value precompile_attributes(instruction_ref ins)
{
    return ins->get_operator().to_value().at("op").at("operator");
}

// The precompile_op with other attributes of the operator it wraps
static operation with_precompile_attributes(instruction_ref ins, const value& attributes)
{
    auto v              = ins->get_operator().to_value();
    v["op"]["operator"] = attributes;
    return make_op("gpu::precompile_op", v);
}

// Fuse what computes the table of matrices of a gather_dot into it: the
// views become the layout of the matrices in the tensor they read, a
// pointwise operator the weights submodule, and the bytes of an unpack_int4
// are read packed, so the kernel computes the matrices from the stored
// weights as it reads them.
struct find_gather_dot_weights
{
    auto matcher() const { return precompile_name("gpu::gather_dot"); }

    static bool is_view(instruction_ref ins)
    {
        return contains({"reshape_lazy", "transpose", "squeeze", "unsqueeze"}, ins->name());
    }

    static bool is_precompile(instruction_ref ins, const std::string& name)
    {
        return ins->name() == "gpu::precompile_op" and
               ins->get_operator().to_value().at("op").at("name").to<std::string>() == name;
    }

    // The packed bytes of an input that is unpacked along its last axis,
    // viewed with half the last axis of the input
    static optional<instruction_ref>
    find_packed(module& m, instruction_ref pos, instruction_ref input)
    {
        if(not input->get_shape().standard())
            return nullopt;
        // the views of a standard tensor that keep it standard keep its order
        auto unpack = input;
        while(contains({"reshape_lazy", "squeeze", "unsqueeze"}, unpack->name()))
            unpack = unpack->inputs().front();
        if(not is_precompile(unpack, "unpack_int4") or not unpack->get_shape().standard())
            return nullopt;
        auto bytes     = unpack->inputs().front();
        const auto& bs = bytes->get_shape();
        auto axis      = precompile_attributes(unpack).at("axis").to<std::int64_t>();
        auto lens      = input->get_shape().lens();
        if(axis < 0)
            axis += bs.ndim();
        if(not bs.standard() or axis + 1 != bs.ndim() or lens.back() % 2 != 0 or
           input->get_shape().elements() != 2 * bs.elements())
            return nullopt;
        lens.back() /= 2;
        if(lens == bs.lens())
            return bytes;
        return m.insert_instruction(pos, make_op("reshape_lazy", {{"dims", lens}}), bytes);
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto attrs = precompile_attributes(ins);
        // only the matrices as the pass left them
        if(not attrs.at("modules").empty() or attrs.at("weights").to<std::size_t>() != 1 or
           not from_value<shape>(attrs.at("layout")).standard())
            return;
        auto weights = ins->inputs().at(2);
        auto base    = weights;
        bool shared  = false;
        while(is_view(base))
        {
            shared = shared or base->outputs().size() > 1;
            base   = base->inputs().front();
        }
        // The shape of the views is where the matrices are in a standard base
        if(not base->get_shape().standard() or weights->get_shape().broadcasted())
            return;
        const auto& base_inputs = base->inputs();
        const bool fuse =
            is_precompile(base, "pointwise") and not shared and base->outputs().size() == 1 and
            base->get_shape().type() != shape::tuple_type and
            std::none_of(base_inputs.begin(), base_inputs.end(), [](instruction_ref input) {
                return input->get_shape().type() == shape::bool_type;
            });
        if(not fuse and base == weights)
            return;
        std::vector<instruction_ref> inputs = {ins->inputs()[0], ins->inputs()[1]};
        std::vector<std::size_t> packed;
        std::vector<module_ref> mods;
        if(fuse)
        {
            // the last input of the pointwise is its output buffer
            std::for_each(base_inputs.begin(), std::prev(base_inputs.end()), [&](auto input) {
                auto bytes = find_packed(m, ins, input);
                if(bytes.has_value())
                    packed.push_back(inputs.size() - 2);
                inputs.push_back(bytes.has_value() ? *bytes : input);
            });
            mods             = base->module_inputs();
            attrs["modules"] = to_value(std::vector<std::string>{"weights"});
        }
        else
        {
            inputs.push_back(base);
        }
        attrs["layout"]  = to_value(weights->get_shape());
        attrs["packed"]  = to_value(packed);
        attrs["weights"] = inputs.size() - 2;
        // the bias and the output buffer follow the one weight tensor
        inputs.insert(inputs.end(), ins->inputs().begin() + 3, ins->inputs().end());
        m.replace_instruction(ins, with_precompile_attributes(ins, attrs), inputs, mods);
    }
};

// The other inputs of a pointwise operator fused in a gather_dot. An input
// gathering its rows with the indices of the gather_dot becomes the table it
// gathers from, which the kernel reads by the entry of the row.
struct gather_dot_inputs
{
    std::vector<instruction_ref> inputs = {};
    // the inputs that are tables
    std::vector<std::size_t> tables = {};

    static optional<instruction_ref> find_table(instruction_ref dot, instruction_ref input)
    {
        if(not input->get_shape().standard())
            return nullopt;
        // the views of a standard tensor that keep it standard keep its order
        auto gather = input;
        while(contains({"reshape_lazy", "squeeze", "unsqueeze"}, gather->name()))
            gather = gather->inputs().front();
        if(not find_gather_dot_weights::is_precompile(gather, "gather") or
           not gather->get_shape().standard())
            return nullopt;
        auto table       = gather->inputs().front();
        auto indices     = gather->inputs().at(1);
        auto dot_indices = dot->inputs().at(1);
        const auto& lens = table->get_shape().lens();
        auto axis        = precompile_attributes(gather).at("axis").to<std::int64_t>();
        auto entries = from_value<shape>(precompile_attributes(dot).at("layout")).lens().front();
        if(axis < 0)
            axis += lens.size();
        if(axis != 0 or lens.front() != entries or indices != dot_indices)
            return nullopt;
        return table;
    }

    void add(instruction_ref dot, instruction_ref input)
    {
        auto table = find_table(dot, input);
        if(table.has_value())
            tables.push_back(inputs.size());
        inputs.push_back(table.has_value() ? *table : input);
    }
};

// Fuse a pointwise operator applied to the sums of a gather_dot, eg an
// activation, as its column submodule
struct find_gather_dot_pointwise
{
    auto matcher() const
    {
        return precompile_name("pointwise")(
            match::not_tuple(),
            match::any_of[match::inputs()](
                precompile_name("gpu::gather_dot")(match::used_once()).bind("dot")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto pw      = r.result;
        auto dot     = r.instructions["dot"];
        auto attrs   = precompile_attributes(dot);
        auto modules = attrs.at("modules").to_vector<std::string>();
        if(contains(modules, "column") or contains(modules, "group"))
            return;
        if(pw->get_shape().lens() != dot->get_shape().lens())
            return;
        auto pw_inputs = pw->inputs();
        if(std::count(pw_inputs.begin(), pw_inputs.end(), dot) != 1)
            return;
        auto pos            = std::find(pw_inputs.begin(), pw_inputs.end(), dot);
        attrs["column_dot"] = std::distance(pw_inputs.begin(), pos);
        pw_inputs.erase(pos);
        // the other inputs, then the output buffer
        gather_dot_inputs others;
        std::for_each(pw_inputs.begin(), std::prev(pw_inputs.end()), [&](auto input) {
            others.add(dot, input);
        });
        attrs["column_inputs"] = others.inputs.size();
        attrs["column_tables"] = to_value(others.tables);
        modules.emplace_back("column");
        attrs["modules"] = to_value(modules);
        auto inputs      = dot->inputs();
        inputs.pop_back();
        inputs.insert(inputs.end(), others.inputs.begin(), others.inputs.end());
        inputs.push_back(pw_inputs.back());
        auto mods = dot->module_inputs();
        mods.push_back(pw->module_inputs().front());
        auto v            = with_precompile_attributes(dot, attrs).to_value();
        v["output_shape"] = to_value(pw->get_shape());
        m.replace_instruction(pw, make_op("gpu::precompile_op", v), inputs, mods);
    }
};

// Fuse a pointwise operator that combines adjacent columns of the sums of a
// gather_dot, eg a swiglu over interleaved gate and linear columns, as its
// group submodule: the pointwise reads slices of the last axis of a view of
// the sums that splits the columns in groups.
struct find_gather_dot_group
{
    auto matcher() const { return precompile_name("pointwise")(match::not_tuple()); }

    struct column_slice
    {
        instruction_ref dot;
        instruction_ref view;
        instruction_ref slice;
        std::size_t column = 0;
    };

    // The column of the groups an input reads, if it is one
    static optional<column_slice> find_column(instruction_ref input)
    {
        auto slice = input;
        while(contains({"squeeze", "unsqueeze"}, slice->name()) and slice->outputs().size() == 1)
            slice = slice->inputs().front();
        if(slice->name() != "slice" or slice->outputs().size() != 1)
            return nullopt;
        auto v         = slice->get_operator().to_value();
        auto axes      = v.at("axes").to_vector<std::size_t>();
        auto starts    = v.at("starts").to_vector<std::size_t>();
        auto ends      = v.at("ends").to_vector<std::size_t>();
        auto view      = slice->inputs().front();
        const auto& vs = view->get_shape();
        if(axes.size() != 1 or axes.front() + 1 != vs.ndim() or ends.front() != starts.front() + 1)
            return nullopt;
        // the views of the sums that keep them standard keep their order
        if(not vs.standard())
            return nullopt;
        auto dot = view;
        while(contains({"reshape_lazy", "squeeze", "unsqueeze"}, dot->name()) and
              dot->inputs().front()->outputs().size() == 1)
            dot = dot->inputs().front();
        if(not find_gather_dot_weights::is_precompile(dot, "gpu::gather_dot"))
            return nullopt;
        return column_slice{dot, view, slice, starts.front()};
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto pw        = r.result;
        auto pw_inputs = pw->inputs();
        pw_inputs.pop_back();
        std::vector<optional<column_slice>> columns;
        std::transform(
            pw_inputs.begin(), pw_inputs.end(), std::back_inserter(columns), &find_column);
        auto first = std::find_if(
            columns.begin(), columns.end(), [](const auto& c) { return c.has_value(); });
        if(first == columns.end())
            return;
        auto dot  = (*first)->dot;
        auto view = (*first)->view;
        if(not std::all_of(columns.begin(), columns.end(), [&](const auto& c) {
               return not c.has_value() or (c->dot == dot and c->view == view);
           }))
            return;
        // the pointwise is the only reader of the sums
        if(not std::all_of(view->outputs().begin(), view->outputs().end(), [&](auto out) {
               return std::any_of(columns.begin(), columns.end(), [&](const auto& c) {
                   return c.has_value() and c->slice == out;
               });
           }))
            return;
        auto attrs   = precompile_attributes(dot);
        auto modules = attrs.at("modules").to_vector<std::string>();
        auto group   = view->get_shape().lens().back();
        auto n       = from_value<shape>(attrs.at("layout")).lens().at(2);
        // the kernel reads the columns of a group from adjacent lanes of a
        // block of 16 columns
        constexpr std::size_t block = 16;
        if(contains(modules, "group") or group < 2 or group > block or
           std::bitset<64>(group).count() != 1 or n % group != 0 or
           pw->get_shape().elements() * group != dot->get_shape().elements())
            return;
        std::vector<std::int64_t> group_args;
        std::transform(
            columns.begin(),
            columns.end(),
            std::back_inserter(group_args),
            [](const auto& c) -> std::int64_t { return c.has_value() ? c->column : -1; });
        // the inputs that are not columns follow, then the output buffer
        gather_dot_inputs others;
        migraphx::for_each(
            pw_inputs.begin(), pw_inputs.end(), columns.begin(), [&](auto input, const auto& c) {
                if(not c.has_value())
                    others.add(dot, input);
            });
        auto inputs = dot->inputs();
        inputs.pop_back();
        inputs.insert(inputs.end(), others.inputs.begin(), others.inputs.end());
        inputs.push_back(pw->inputs().back());
        attrs["group"]        = group;
        attrs["group_args"]   = to_value(group_args);
        attrs["group_tables"] = to_value(others.tables);
        modules.emplace_back("group");
        attrs["modules"] = to_value(modules);
        auto mods        = dot->module_inputs();
        mods.push_back(pw->module_inputs().front());
        auto v            = with_precompile_attributes(dot, attrs).to_value();
        v["output_shape"] = to_value(pw->get_shape());
        m.replace_instruction(pw, make_op("gpu::precompile_op", v), inputs, mods);
    }
};

void fuse_ops::apply(module& m) const
{
    match::find_matches(m,
                        find_pointwise_layout_contiguous{},
                        find_contiguous_layout_pointwise{},
                        find_precompile_copy{});
    run_passes(m, {dead_code_elimination{}});
#if MIGRAPHX_USE_MIOPEN
    match::find_matches(m, find_conv_pointwise{ctx}, find_conv_bias_relu{ctx}, find_conv_bias{ctx});
    run_passes(m, {dead_code_elimination{}});
#endif
#if MIGRAPHX_USE_ROCBLAS
    match::find_matches(m, find_rocblas_gemm_pointwise{});
#endif
#if MIGRAPHX_USE_HIPBLASLT
    match::find_matches(m, find_hipblas_gemm_pointwise{});
#endif
    match::find_matches(m,
                        find_layernorm_pointwise{},
                        find_channelwise_conv_pointwise{},
                        find_winograd_conv_pointwise{},
                        find_concat_pointwise{},
                        find_contiguous_transpose_rocblas_gemm{},
#if MIGRAPHX_USE_HIPBLASLT
                        find_contiguous_transpose_hip_gemm{},
#endif
                        find_commutative_broadcast{});
    // In the order of the instructions: the weights of a gather_dot, then the
    // pointwise operators reading its sums
    match::find_matches(
        m, find_gather_dot_weights{}, find_gather_dot_pointwise{}, find_gather_dot_group{});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
