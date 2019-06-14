#include <migraphx/gpu/fuse_ops.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <migraphx/gpu/convolution.hpp>
#include <migraphx/gpu/device/add_relu.hpp>
#include <migraphx/gpu/device/add.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

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

    fusion(const shape& input)
    // : fp(make_fusion_plan(input))
    {
        auto t = make_tensor(input);
        fp     = make_fusion_plan(t);
        keep_alive(std::move(t));
    }

    op_t operator[](std::size_t i) const
    {
        op_t result;
        auto status = miopenFusionPlanGetOp(fp.get(), i, &result);
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Failed retrieving operator at " + std::to_string(i));
        return result;
    }

    auto get() const { return fp.get(); }

    op_t create_bias(const shape& bias)
    {
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
        op_t result;
        auto status = miopenCreateOpActivationForward(fp.get(), &result, miopenActivationRELU);
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Creating operator failed");
        return result;
    }

    op_t create_conv(const op::convolution& op, const shape& weights)
    {
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
        // TODO: Use zero workspace for now
        std::size_t ws_size = 0;
        // int algo_count = 1;
        // miopenConvFwdAlgorithm_t algo;
        // miopenFusionPlanConvolutionGetAlgo(fp.get(), 1, &algo_count, &algo);
        // miopenFusionPlanGetWorkSpaceSize(ctx.get_stream().get_miopen(), fp.get(), &ws_size,
        // algo);
        return shape{shape::int8_type, {ws_size}};
    }

    void compile(context& ctx)
    {
        auto status = miopenCompileFusionPlan(ctx.get_stream().get_miopen(), fp.get());
        if(status != miopenStatusSuccess)
            MIGRAPHX_THROW("Compiling fusion plan failed");
    }

    argument execute(context& ctx,
                     const fused_operator_args& fargs,
                     const argument& x,
                     const argument& y) const
    {
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

MIGRAPHX_PRED_MATCHER(bias_shape, instruction_ref ins)
{
    auto&& s = ins->get_shape();
    return s.broadcasted() and s.strides().size() == 4 and s.strides()[0] == 0 and
           s.strides()[1] != 0 and s.strides()[2] == 0 and s.strides()[3] == 0;
}

// TODO: Move to another header
template <class T, class... Ts>
std::array<T, sizeof...(Ts) + 1> make_array(T x, Ts... xs)
{
    return {std::move(x), std::move(static_cast<T>(xs))...};
}

MIGRAPHX_PRED_MATCHER(fusable_conv, instruction_ref ins)
{
    if(ins->name() != "gpu::convolution")
        return false;
    if(ins->get_shape().type() != shape::float_type)
        return false;
    auto wei = ins->inputs().at(1)->get_shape();
    assert(wei.lens().size() == 4);
    auto conv = any_cast<miopen_convolution>(ins->get_operator());
    if(conv.op.group > 1)
        return false;
    if(conv.op.padding_mode != op::padding_mode_t::default_)
        return false;
    if(wei.lens()[1] > 512 and conv.algo != miopenConvolutionFwdAlgoWinograd)
        return false;
    auto op = conv.op;
    return contains({{0, 0}, {1, 1}, {2, 2}}, op.padding) and
           contains({{0, 0}, {1, 1}}, op.stride) and op.dilation == make_array<size_t>(1, 1);
}

struct hip_triadd
{
    std::string name() const { return "hip::triadd"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(4);
        return inputs.front();
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        device::add(ctx.get_stream().get(), args.at(3), args.at(0), args.at(1), args.at(2));
        return args.at(3);
    }
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

struct hip_triadd_relu
{
    std::string name() const { return "hip::triadd_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(4);
        return inputs.front();
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        device::add_relu(ctx.get_stream().get(), args.at(3), args.at(0), args.at(1), args.at(2));
        return args.at(3);
    }
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

struct hip_add_relu
{
    std::string name() const { return "hip::add_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return inputs.front();
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        device::add_relu(ctx.get_stream().get(), args.at(2), args.at(0), args.at(1));
        return args.at(2);
    }
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

struct find_add_relu
{
    auto matcher() const
    {
        return match::name("gpu::relu")(match::arg(0)(
            match::any_of(match::name("gpu::add"), match::name("hip::triadd")).bind("add")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto add_ins = r.instructions["add"];
        auto ins     = r.result;
        auto args    = add_ins->inputs();
        // Use the allocation from the relu operator
        args.back() = ins->inputs().back();
        if(add_ins->name() == "gpu::add")
            p.replace_instruction(ins, hip_add_relu{}, args);
        else if(add_ins->name() == "hip::triadd")
            p.replace_instruction(ins, hip_triadd_relu{}, args);
    }
};

struct find_triadd
{
    auto matcher() const
    {
        return match::name("gpu::add")(match::either_arg(0, 1)(match::name("gpu::add").bind("add"),
                                                               match::any().bind("input")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto add_ins        = r.instructions["add"];
        auto input_ins      = r.instructions["input"];
        auto ins            = r.result;
        auto args           = add_ins->inputs();
        auto is_broadcasted = [](auto arg) { return arg->get_shape().broadcasted(); };
        if(std::count_if(args.begin(), args.end(), is_broadcasted) > 1)
            return;
        args.insert(args.begin(), input_ins);
        // Ensure the last arguments is the broadcasted one
        auto it = std::find_if(args.begin(), args.end(), is_broadcasted);
        if(it != args.end())
            std::swap(*it, *std::prev(args.end(), 2));
        args.back() = ins->inputs().back();
        p.replace_instruction(ins, hip_triadd{}, args);
    }
};

struct miopen_conv_bias
{
    op::convolution op;
    fusion f;
    fusion::op_t conv;
    fusion::op_t bias;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return op::convolution::reflect(self.op, f);
    }

    miopen_conv_bias(op::convolution c, const shape& input, const shape& weights, const shape& b)
        : op(c), f(input)
    {
        conv = f.create_conv(op, weights);
        bias = f.create_bias(b);
    }

    std::string name() const { return "gpu::conv_bias"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        // TODO: Check slices
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        auto fargs  = make_fused_args();
        float alpha = 1;
        float beta  = 0;
        miopenSetOpArgsConvForward(fargs.get(), conv, &alpha, &beta, args[1].implicit());
        miopenSetOpArgsBiasForward(fargs.get(), bias, &alpha, &beta, args[3].implicit());
        return f.execute(ctx, fargs, args[0], args[4]);
    }

    void finalize(context& ctx, const shape&, const std::vector<shape>&) { f.compile(ctx); }
    shape get_workspace(context& ctx) { return f.get_workspace(ctx); }
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

struct miopen_conv_bias_relu
{
    op::convolution op;
    fusion f;
    fusion::op_t conv;
    fusion::op_t bias;
    fusion::op_t relu;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return op::convolution::reflect(self.op, f);
    }

    miopen_conv_bias_relu(op::convolution c,
                          const shape& input,
                          const shape& weights,
                          const shape& b)
        : op(c), f(input)
    {
        conv = f.create_conv(op, weights);
        bias = f.create_bias(b);
        relu = f.create_relu();
    }

    std::string name() const { return "gpu::conv_bias_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        // TODO: Check slices
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        auto fargs  = make_fused_args();
        float alpha = 1;
        float beta  = 0;
        miopenSetOpArgsConvForward(fargs.get(), conv, &alpha, &beta, args[1].implicit());
        miopenSetOpArgsBiasForward(fargs.get(), bias, &alpha, &beta, args[3].implicit());
        miopenSetOpArgsActivForward(fargs.get(), relu, &alpha, &beta, 0, 0, 0);
        return f.execute(ctx, fargs, args[0], args[4]);
    }
    void finalize(context& ctx, const shape&, const std::vector<shape>&) { f.compile(ctx); }
    shape get_workspace(context& ctx) { return f.get_workspace(ctx); }
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

template <class... Ms>
auto conv_bias(Ms... ms)
{
    return match::name("gpu::add")(
        match::either_arg(0, 1)(bias_shape(match::used_once()).bind("bias"),
                                fusable_conv(match::used_once()).bind("conv")),
        ms...);
}

template <class Op>
void apply_conv_bias(context& ctx, program& p, match::matcher_result r)
{
    auto conv_ins    = r.instructions["conv"];
    auto bias_ins    = r.instructions["bias"];
    auto ins         = r.result;
    auto input_ins   = conv_ins->inputs().at(0);
    auto weights_ins = conv_ins->inputs().at(1);
    auto conv_op     = any_cast<miopen_convolution>(conv_ins->get_operator()).op;
    auto alloc_ins   = ins->inputs().back();
    auto old_ws_ins  = conv_ins->inputs().at(2);

    Op cb{conv_op, input_ins->get_shape(), weights_ins->get_shape(), bias_ins->get_shape()};
    // TODO: Insert ws allocation
    auto ws = cb.get_workspace(ctx);
    (void)ws;
    p.replace_instruction(ins, cb, input_ins, weights_ins, old_ws_ins, bias_ins, alloc_ins);
}

struct find_conv_bias
{
    context* ctx = nullptr;
    auto matcher() const
    {
        return conv_bias(match::none_of(match::output(match::name("gpu::relu"))));
    }

    void apply(program& p, match::matcher_result r) const
    {
        apply_conv_bias<miopen_conv_bias>(*ctx, p, std::move(r));
    }
};

struct find_conv_bias_relu
{
    context* ctx = nullptr;
    auto matcher() const { return match::name("gpu::relu")(match::arg(0)(conv_bias())); }

    void apply(program& p, match::matcher_result r) const
    {
        apply_conv_bias<miopen_conv_bias_relu>(*ctx, p, std::move(r));
    }
};

void fuse_ops::apply(program& p) const
{
    // clang-format off
    match::find_matches(p, find_triadd{});
    match::find_matches(p, 
        find_conv_bias_relu{ctx},
        find_conv_bias{ctx},
        find_add_relu{}
    );
    // clang-format on
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
