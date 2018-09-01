#include <rocblas.h>
#include <migraph/gpu/lowering.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
#include <migraph/shape_for_each.hpp>
#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/dfor.hpp>
#include <migraph/gpu/device/contiguous.hpp>
#include <migraph/gpu/device/add.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/rocblas.hpp>
#include <migraph/gpu/context.hpp>
#include <utility>

namespace migraph {
namespace gpu {

struct miopen_batch_norm_inference
{
    batch_norm_inference op;

    std::string name() const { return "gpu::batch_norm_inference"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(6);
        return op.compute_shape(
            {inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3), inputs.at(4)});
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        auto x_desc  = make_tensor(args[0].get_shape());
        auto y_desc  = make_tensor(output_shape);
        auto bn_desc = make_tensor(args[3].get_shape());

        float alpha = 1.0, beta = 0.0f;

        miopenBatchNormalizationForwardInference(ctx.handle.get(),
                                                 miopenBatchNormMode_t(op.bn_mode),
                                                 &alpha,
                                                 &beta,
                                                 x_desc.get(),
                                                 args[0].implicit(),
                                                 y_desc.get(),
                                                 args[5].implicit(),
                                                 bn_desc.get(),
                                                 args[1].implicit(),
                                                 args[2].implicit(),
                                                 args[3].implicit(),
                                                 args[4].implicit(),
                                                 op.epsilon);

        return args[5];
    }
};

struct miopen_convolution
{
    convolution op;
    shared<convolution_descriptor> cd;
    miopenConvFwdAlgorithm_t algo{};

    std::string name() const { return "gpu::convolution"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(4).standard();
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        auto x_desc = make_tensor(args[0].get_shape());
        auto w_desc = make_tensor(args[1].get_shape());
        auto y_desc = make_tensor(output_shape);

        float alpha = 1, beta = 0;
        miopenConvolutionForward(ctx.handle.get(),
                                 &alpha,
                                 x_desc.get(),
                                 args[0].implicit(),
                                 w_desc.get(),
                                 args[1].implicit(),
                                 cd.get(),
                                 algo,
                                 &beta,
                                 y_desc.get(),
                                 args[3].implicit(),
                                 args[2].implicit(),
                                 args[2].get_shape().bytes());
        return args[3];
    }

    shape compile(context& ctx, const shape& output_shape, std::vector<instruction_ref> inputs)
    {
        shape workspace_shape{};
        auto x_desc = make_tensor(inputs[0]->get_shape());
        auto w_desc = make_tensor(inputs[1]->get_shape());
        auto y_desc = make_tensor(output_shape);

        std::size_t workspace_size = 0;
        miopenConvolutionForwardGetWorkSpaceSize(
            ctx.handle.get(), w_desc.get(), x_desc.get(), cd.get(), y_desc.get(), &workspace_size);
        workspace_shape = shape{shape::int8_type, {workspace_size}};

        auto x         = to_gpu(generate_argument(inputs[0]->get_shape()));
        auto w         = to_gpu(generate_argument(inputs[1]->get_shape()));
        auto y         = to_gpu(generate_argument(output_shape));
        auto workspace = allocate_gpu(workspace_shape);

        int algo_count = 1;
        miopenConvAlgoPerf_t perf;
        miopenFindConvolutionForwardAlgorithm(ctx.handle.get(),
                                              x_desc.get(),
                                              x.implicit(),
                                              w_desc.get(),
                                              w.implicit(),
                                              cd.get(),
                                              y_desc.get(),
                                              y.implicit(),
                                              1,
                                              &algo_count,
                                              &perf,
                                              workspace.implicit(),
                                              workspace_size,
                                              false);
        algo = perf.fwd_algo;
        return algo == miopenConvolutionFwdAlgoWinograd ? shape{shape::int8_type, {0}}
                                                        : workspace_shape;
    }
};

struct miopen_pooling
{
    pooling op;
    shared<pooling_descriptor> pd;

    std::string name() const { return "gpu::pooling"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2).standard();
        return op.compute_shape({inputs.at(0)});
    }
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        auto x_desc = make_tensor(args[0].get_shape());
        auto y_desc = make_tensor(output_shape);

        float alpha = 1, beta = 0;

        miopenPoolingForward(ctx.handle.get(),
                             pd.get(),
                             &alpha,
                             x_desc.get(),
                             args[0].implicit(),
                             &beta,
                             y_desc.get(),
                             args[1].implicit(),
                             false,
                             nullptr,
                             0);

        return args[1];
    }
};

struct hip_add
{
    std::string name() const { return "gpu::add"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        // check_shapes{inputs, *this}.has(3).standard();
        check_shapes{inputs, *this}.has(3);
        return inputs.at(0);
    }

    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        device::add(args[2], args[0], args[1]);
        return args[2];
    }
};

struct miopen_add
{
    std::string name() const { return "gpu::add"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(3).not_broadcasted();
        return inputs.at(0);
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        float alpha = 1, beta = 0;
        auto a_desc = make_tensor(args[0].get_shape());
        auto b_desc = make_tensor(args[1].get_shape());
        auto c_desc = make_tensor(output_shape);
        miopenOpTensor(ctx.handle.get(),
                       miopenTensorOpAdd,
                       &alpha,
                       a_desc.get(),
                       args[0].implicit(),
                       &alpha,
                       b_desc.get(),
                       args[1].implicit(),
                       &beta,
                       c_desc.get(),
                       args[2].implicit());
        return args[2];
    }
};

struct miopen_gemm
{
    gemm op;
    std::string name() const { return "gpu::gemm"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        float alpha     = 1.0f;
        float beta      = 0.0f;
        bool transa     = args[0].get_shape().transposed();
        bool transb     = args[1].get_shape().transposed();
        rocblas_int lda = args[0].get_shape().strides()[transa ? 1 : 0];
        rocblas_int ldb = args[1].get_shape().strides()[transb ? 1 : 0];
        rocblas_int ldc = args[2].get_shape().strides()[0];
        rocblas_int m   = output_shape.lens()[0];
        rocblas_int n   = output_shape.lens()[1];
        rocblas_int k   = args[0].get_shape().lens()[1];
        rocblas_sgemm(ctx.rbhandle.get(),
                      transb ? rocblas_operation_transpose : rocblas_operation_none,
                      transa ? rocblas_operation_transpose : rocblas_operation_none,
                      n,
                      m,
                      k,
                      &alpha,
                      args[1].implicit(),
                      ldb,
                      args[0].implicit(),
                      lda,
                      &beta,
                      args[2].implicit(),
                      ldc);
        return args[2];
    }
};

struct miopen_contiguous
{
    contiguous op;
    std::string name() const { return "gpu::contiguous"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return op.compute_shape({inputs.at(0)});
    }
    argument compute(context&, shape output_shape, const std::vector<argument>& args) const
    {
        assert(output_shape == args[1].get_shape());
        assert(output_shape.standard());
        (void)output_shape;
        device::contiguous(args.at(1), args.at(0));
        return args.at(1);
    }
};

struct miopen_relu
{
    shared<activation_descriptor> ad;
    std::string name() const { return "gpu::relu"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2).not_broadcasted();
        return inputs.at(1);
    }

    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        float alpha = 1, beta = 0;
        auto x_desc = make_tensor(args[0].get_shape());
        auto y_desc = make_tensor(output_shape);
        miopenActivationForward(ctx.handle.get(),
                                ad.get(),
                                &alpha,
                                x_desc.get(),
                                args[0].implicit(),
                                &beta,
                                y_desc.get(),
                                args[1].implicit());

        return args[1];
    }
};

struct miopen_apply
{
    program* prog = nullptr;
    context ctx{};

    void check_shape(shape x, instruction_ref i)
    {
        assert(x == i->get_shape());
        (void)x;
        (void)i;
    }

    void apply()
    {
        for(auto it = prog->begin(); it != prog->end(); it++)
        {
            auto s = it->get_shape();
            if(it->op.name() == "convolution")
            {
                check_shape(s, apply_convolution(it));
            }
            else if(it->op.name() == "activation")
            {
                check_shape(s, apply_activation(it));
            }
            else if(it->op.name() == "pooling")
            {
                check_shape(s, apply_pooling(it));
            }
            else if(it->op.name() == "add")
            {
                check_shape(s, apply_add(it));
            }
            else if(it->op.name() == "gemm")
            {
                check_shape(s, apply_gemm(it));
            }
            else if(it->op.name() == "contiguous")
            {
                check_shape(s, apply_contiguous(it));
            }
            else if(it->op.name() == "batch_norm_inference")
            {
                check_shape(s, apply_batch_norm_inference(it));
            }
        }
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s, std::string tag = "")
    {
        if(ins == --prog->end() and not tag.empty())
        {
            return prog->add_parameter("output", s);
        }
        else
        {
            auto is     = prog->add_outline(s);
            auto result = prog->insert_instruction(ins, hip_allocate{std::move(tag)}, is);
            return result;
        }
    }

    instruction_ref apply_convolution(instruction_ref ins)
    {
        auto&& op = any_cast<convolution>(ins->op);

        auto conv = miopen_convolution{op, make_conv(op)};
        auto ws   = conv.compile(ctx, ins->result, ins->arguments);

        auto workspace = insert_allocation(ins, ws, "workspace");
        auto output    = insert_allocation(ins, ins->result);

        return prog->replace_instruction(
            ins, conv, ins->arguments.at(0), ins->arguments.at(1), workspace, output);
    }

    instruction_ref apply_pooling(instruction_ref ins)
    {
        auto&& op   = any_cast<pooling>(ins->op);
        auto pd     = make_pooling(op);
        auto output = insert_allocation(ins, ins->result);

        return prog->replace_instruction(
            ins, miopen_pooling{op, std::move(pd)}, ins->arguments.at(0), output);
    }

    instruction_ref apply_activation(instruction_ref ins)
    {
        auto&& op = any_cast<activation>(ins->op);
        auto ad   = make_relu();
        if(op.mode == "relu")
        {
            auto output = insert_allocation(ins, ins->result);
            return prog->replace_instruction(
                ins, miopen_relu{std::move(ad)}, ins->arguments.at(0), output);
        }
        return ins;
    }

    instruction_ref apply_add(instruction_ref ins)
    {
        auto output = insert_allocation(ins, ins->result);
        return prog->replace_instruction(
            ins, hip_add{}, ins->arguments.at(0), ins->arguments.at(1), output);
    }

    instruction_ref apply_gemm(instruction_ref ins)
    {
        auto&& op   = any_cast<gemm>(ins->op);
        auto output = insert_allocation(ins, ins->result);
        return prog->replace_instruction(
            ins, miopen_gemm{op}, ins->arguments.at(0), ins->arguments.at(1), output);
    }

    instruction_ref apply_contiguous(instruction_ref ins)
    {
        auto&& op   = any_cast<contiguous>(ins->op);
        auto output = insert_allocation(ins, ins->result);
        return prog->replace_instruction(ins, miopen_contiguous{op}, ins->arguments.at(0), output);
    }

    instruction_ref apply_batch_norm_inference(instruction_ref ins)
    {
        auto&& op       = any_cast<batch_norm_inference>(ins->op);
        auto output     = insert_allocation(ins, ins->result);
        shape old_shape = ins->arguments.at(1)->get_shape();
        std::vector<int64_t> new_shape{1, static_cast<int64_t>(old_shape.elements()), 1, 1};
        auto reshape_op = reshape{new_shape};
        std::vector<instruction_ref> reshapes;
        std::transform(ins->arguments.begin() + 1,
                       ins->arguments.end(),
                       std::back_inserter(reshapes),
                       [&](auto i) { return prog->insert_instruction(ins, reshape_op, i); });
        return prog->replace_instruction(ins,
                                         miopen_batch_norm_inference{op},
                                         ins->arguments.at(0),
                                         reshapes[0],
                                         reshapes[1],
                                         reshapes[2],
                                         reshapes[3],
                                         output);
    }
};

void lowering::apply(program& p) const { miopen_apply{&p, ctx}.apply(); }

} // namespace gpu

} // namespace migraph
