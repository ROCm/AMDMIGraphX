#include <migraph/gpu/fuse_ops.hpp>
#include <migraph/matcher.hpp>
#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/convolution.hpp>
#include <migraph/gpu/device/add_relu.hpp>
#include <migraph/instruction.hpp>

namespace migraph {

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
            MIGRAPH_THROW("Failed retrieving operator at " + std::to_string(i));
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
            MIGRAPH_THROW("Creating operator failed");
        return result;
    }

    op_t create_relu()
    {
        op_t result;
        auto status = miopenCreateOpActivationForward(fp.get(), &result, miopenActivationRELU);
        if(status != miopenStatusSuccess)
            MIGRAPH_THROW("Creating operator failed");
        return result;
    }

    op_t create_conv(const op::convolution& op, const shape& weights)
    {
        op_t result;
        auto cd     = keep_alive(make_conv(op));
        auto t      = keep_alive(make_tensor(weights));
        auto status = miopenCreateOpConvForward(fp.get(), &result, cd.get(), t.get());
        if(status != miopenStatusSuccess)
            MIGRAPH_THROW("Creating operator failed");
        return result;
    }
};

MIGRAPH_PRED_MATCHER(bias_shape, instruction_ref ins)
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

MIGRAPH_PRED_MATCHER(fusable_conv, instruction_ref ins)
{
    if(ins->name() != "gpu::convolution")
        return false;
    auto op = any_cast<miopen_convolution>(ins->get_operator()).op;
    return op.padding == make_array<size_t>(0, 0) and op.stride == make_array<size_t>(1, 1) and
           op.dilation == make_array<size_t>(1, 1);
}

struct hip_add_relu
{
    std::string name() const { return "hip::add_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return inputs.front();
    }
    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        device::add_relu(args.at(2), args.at(0), args.at(1));
        return args.at(2);
    }
};

struct match_add_relu
{
    auto matcher() const
    {
        return match::name("gpu::relu")(match::arg(0)(match::name("gpu::add").bind("add")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto add_ins = r.instructions["add"];
        auto ins     = r.result;
        auto args    = add_ins->inputs();
        // Use the allocation from the relu operator
        args.back() = ins->inputs().back();
        p.replace_instruction(ins, hip_add_relu{}, args);
    }
};

struct miopen_conv_bias
{
    op::convolution op;
    fusion f;
    fusion::op_t conv;
    fusion::op_t bias;

    miopen_conv_bias(op::convolution c, shape input, shape weights, shape b) : op(c), f(input)
    {
        f.create_conv(op, weights);
        f.create_bias(b);
    }

    std::string name() const { return "gpu::conv_bias"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        // TODO: Check slices
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        auto fargs  = make_fused_args();
        float alpha = 1, beta = 0;
        auto x = make_tensor(args[0].get_shape());
        auto y = make_tensor(output_shape);
        miopenSetOpArgsConvForward(fargs.get(), conv, &alpha, &beta, args[1].implicit());
        miopenSetOpArgsBiasForward(fargs.get(), bias, &alpha, &beta, args[3].implicit());
        miopenExecuteFusionPlan(ctx.handle.get(),
                                f.get(),
                                x.get(),
                                args[0].implicit(),
                                y.get(),
                                args[4].implicit(),
                                fargs.get());
        return args.at(4);
    }

    shape compile(context& ctx)
    {
        int algo_count = 1;
        miopenConvFwdAlgorithm_t algo;
        miopenFusionPlanConvolutionGetAlgo(f.get(), 1, &algo_count, &algo);
        std::size_t ws_size = 0;
        miopenFusionPlanGetWorkSpaceSize(ctx.handle.get(), f.get(), &ws_size, algo);
        auto status = miopenCompileFusionPlan(ctx.handle.get(), f.get());
        if(status != miopenStatusSuccess)
            MIGRAPH_THROW("Compiling fusion plan failed");
        return shape{shape::int8_type, {ws_size}};
    }
};

struct miopen_conv_bias_relu
{
    op::convolution op;
    fusion f;
    fusion::op_t conv;
    fusion::op_t bias;

    miopen_conv_bias_relu(op::convolution c, shape input, shape weights, shape b) : op(c), f(input)
    {
        f.create_conv(op, weights);
        f.create_bias(b);
        f.create_relu();
    }

    std::string name() const { return "gpu::conv_bias_relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(5);
        // TODO: Check slices
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        auto fargs  = make_fused_args();
        float alpha = 1, beta = 0;
        auto x = make_tensor(args[0].get_shape());
        auto y = make_tensor(output_shape);
        miopenSetOpArgsConvForward(fargs.get(), conv, &alpha, &beta, args[1].implicit());
        miopenSetOpArgsBiasForward(fargs.get(), bias, &alpha, &beta, args[3].implicit());
        miopenExecuteFusionPlan(ctx.handle.get(),
                                f.get(),
                                x.get(),
                                args[0].implicit(),
                                y.get(),
                                args[4].implicit(),
                                fargs.get());
        return args.at(4);
    }

    shape compile(context& ctx)
    {
        int algo_count = 1;
        miopenConvFwdAlgorithm_t algo;
        miopenFusionPlanConvolutionGetAlgo(f.get(), 1, &algo_count, &algo);
        std::size_t ws_size = 0;
        miopenFusionPlanGetWorkSpaceSize(ctx.handle.get(), f.get(), &ws_size, algo);
        auto status = miopenCompileFusionPlan(ctx.handle.get(), f.get());
        if(status != miopenStatusSuccess)
            MIGRAPH_THROW("Compiling fusion plan failed");
        return shape{shape::int8_type, {ws_size}};
    }
};

template<class... Ms>
auto conv_bias(Ms... ms)
{
    return match::name("gpu::add")(
            match::either_arg(0, 1)(bias_shape().bind("bias"), fusable_conv().bind("conv")), ms...);
}

struct match_conv_bias
{
    context* ctx = nullptr;
    auto matcher() const
    {
        return conv_bias(match::none_of(match::output(match::name("gpu::relu"))));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto conv_ins    = r.instructions["conv"];
        auto bias_ins    = r.instructions["bias"];
        auto ins         = r.result;
        auto input_ins   = conv_ins->inputs().at(0);
        auto weights_ins = conv_ins->inputs().at(1);
        auto conv_op     = any_cast<miopen_convolution>(conv_ins->get_operator()).op;
        auto alloc_ins   = ins->inputs().back();
        auto old_ws_ins  = conv_ins->inputs().at(2);

        miopen_conv_bias cb{
            conv_op, input_ins->get_shape(), weights_ins->get_shape(), bias_ins->get_shape()};
        // TODO: Insert ws allocation
        auto ws = cb.compile(*ctx);

        p.replace_instruction(ins, cb, input_ins, weights_ins, old_ws_ins, bias_ins, alloc_ins);
    }
};

struct match_conv_bias_relu
{
    context* ctx = nullptr;
    auto matcher() const
    {
        return match::name("gpu::relu")(conv_bias());
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto conv_ins    = r.instructions["conv"];
        auto bias_ins    = r.instructions["bias"];
        auto ins         = r.result;
        auto input_ins   = conv_ins->inputs().at(0);
        auto weights_ins = conv_ins->inputs().at(1);
        auto conv_op     = any_cast<miopen_convolution>(conv_ins->get_operator()).op;
        auto alloc_ins   = ins->inputs().back();
        auto old_ws_ins  = conv_ins->inputs().at(2);

        miopen_conv_bias_relu cbr{
            conv_op, input_ins->get_shape(), weights_ins->get_shape(), bias_ins->get_shape()};
        // TODO: Insert ws allocation
        auto ws = cbr.compile(*ctx);

        p.replace_instruction(ins, cbr, input_ins, weights_ins, old_ws_ins, bias_ins, alloc_ins);
    }
};

void fuse_ops::apply(program& p) const
{
    match::find_matches(p, match_add_relu{}, match_conv_bias_relu{ctx}, match_conv_bias{ctx});
}

} // namespace gpu

} // namespace migraph
