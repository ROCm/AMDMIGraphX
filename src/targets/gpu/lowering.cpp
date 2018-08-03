#include <rocblas.h>
#include <migraph/gpu/lowering.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/shape_for_each.hpp>
#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/dfor.hpp>
#include <migraph/gpu/kernels.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/rocblas.hpp>
#include <migraph/gpu/context.hpp>

namespace migraph {
namespace gpu {

struct miopen_convolution
{
    convolution op;
    shared<convolution_descriptor> cd;

    std::string name() const { return "gpu::convolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context& ctx, shape output_shape, std::vector<argument> args) const
    {
        auto x_desc = make_tensor(args[0].get_shape());
        auto w_desc = make_tensor(args[1].get_shape());
        auto y_desc = make_tensor(output_shape);

        float alpha = 1, beta = 0;
        int algo_count;
        miopenConvAlgoPerf_t perf;
        miopenFindConvolutionForwardAlgorithm(ctx.handle.get(),
                                              x_desc.get(),
                                              args[0].implicit(),
                                              w_desc.get(),
                                              args[1].implicit(),
                                              cd.get(),
                                              y_desc.get(),
                                              args[2].implicit(),
                                              1,
                                              &algo_count,
                                              &perf,
                                              nullptr,
                                              0,
                                              false);
        miopenConvolutionForward(ctx.handle.get(),
                                 &alpha,
                                 x_desc.get(),
                                 args[0].implicit(),
                                 w_desc.get(),
                                 args[1].implicit(),
                                 cd.get(),
                                 perf.fwd_algo,
                                 &beta,
                                 y_desc.get(),
                                 args[2].implicit(),
                                 nullptr,
                                 0);
        return args[2];
    }
};

struct miopen_pooling
{
    pooling op;
    shared<pooling_descriptor> pd;

    std::string name() const { return "gpu::pooling"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return op.compute_shape({inputs.at(1)});
    }
    argument compute(context& ctx, shape output_shape, std::vector<argument> args) const
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

struct miopen_add
{
    std::string name() const { return "gpu::add"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return inputs.at(0);
    }

    argument compute(context& ctx, shape output_shape, std::vector<argument> args) const
    {
        if(args[1].get_shape().broadcasted())
        {
            argument result{output_shape};

            visit_all(result, from_gpu(args[0]), from_gpu(args[1]))(
                [&](auto output, auto input1, auto input2) {
                    shape_for_each(output.get_shape(), [&](const auto& idx) {
                        output(idx.begin(), idx.end()) =
                            input1(idx.begin(), idx.end()) + input2(idx.begin(), idx.end());
                    });
                });
            return to_gpu(result);
        }
        else
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
    }
};

struct miopen_gemm
{
    gemm op;
    std::string name() const { return "gpu::convolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context& ctx, shape output_shape, std::vector<argument> args) const
    {
        float alpha     = 1.0f;
        float beta      = 0.0f;
        rocblas_int lda = args[0].get_shape().lens()[1];
        rocblas_int ldb = args[1].get_shape().lens()[1];
        rocblas_int ldc = args[2].get_shape().lens()[1];
        rocblas_int m   = output_shape.lens()[0];
        rocblas_int n   = output_shape.lens()[1];
        rocblas_int k   = args[0].get_shape().lens()[1];
        rocblas_sgemm(ctx.rbhandle.get(),
                      rocblas_operation_none,
                      rocblas_operation_none,
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
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return op.compute_shape({inputs.at(0)});
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        hip_contiguous(output_shape, args.at(0), args.at(1));
        return args.at(1);
    }
};

struct miopen_relu
{
    shared<activation_descriptor> ad;
    std::string name() const { return "gpu::relu"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return inputs.at(1);
    }

    argument compute(context& ctx, shape output_shape, std::vector<argument> args) const
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

    void apply()
    {
        for(auto it = prog->begin(); it != prog->end(); it++)
        {
            if(it->op.name() == "convolution")
            {
                apply_convolution(it);
            }
            else if(it->op.name() == "activation")
            {
                apply_activation(it);
            }
            else if(it->op.name() == "pooling")
            {
                apply_pooling(it);
            }
            else if(it->op.name() == "add")
            {
                apply_add(it);
            }
            else if(it->op.name() == "gemm")
            {
                apply_gemm(it);
            }
            else if(it->op.name() == "contiguous")
            {
                apply_contiguous(it);
            }
        }
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s)
    {
        if(ins == --prog->end())
        {
            return prog->add_parameter("output", s);
        }
        else
        {
            auto is     = prog->add_outline(s);
            auto result = prog->insert_instruction(ins, hip_allocate{}, is);
            return result;
        }
    }

    void apply_convolution(instruction_ref ins)
    {
        auto&& op   = any_cast<convolution>(ins->op);
        auto cd     = make_conv(op);
        auto output = insert_allocation(ins, ins->result);

        prog->replace_instruction(ins,
                                  miopen_convolution{op, std::move(cd)},
                                  ins->arguments.at(0),
                                  ins->arguments.at(1),
                                  output);
    }

    void apply_pooling(instruction_ref ins)
    {
        auto&& op   = any_cast<pooling>(ins->op);
        auto pd     = make_pooling(op);
        auto output = insert_allocation(ins, ins->result);

        prog->replace_instruction(
            ins, miopen_pooling{op, std::move(pd)}, ins->arguments.at(0), output);
    }

    void apply_activation(instruction_ref ins)
    {
        auto&& op = any_cast<activation>(ins->op);
        auto ad   = make_relu();
        if(op.mode == "relu")
        {
            auto output = insert_allocation(ins, ins->result);
            prog->replace_instruction(
                ins, miopen_relu{std::move(ad)}, ins->arguments.at(0), output);
        }
    }

    void apply_add(instruction_ref ins)
    {
        auto output = insert_allocation(ins, ins->result);
        prog->replace_instruction(
            ins, miopen_add{}, ins->arguments.at(0), ins->arguments.at(1), output);
    }

    void apply_gemm(instruction_ref ins)
    {
        auto&& op   = any_cast<gemm>(ins->op);
        auto output = insert_allocation(ins, ins->result);
        prog->replace_instruction(
            ins, miopen_gemm{op}, ins->arguments.at(0), ins->arguments.at(1), output);
    }

    void apply_contiguous(instruction_ref ins)
    {
        auto&& op   = any_cast<contiguous>(ins->op);
        auto output = insert_allocation(ins, ins->result);
        prog->replace_instruction(ins, miopen_contiguous{op}, ins->arguments.at(0), output);
    }
};

void lowering::apply(program& p) const { miopen_apply{&p}.apply(); }

} // namespace gpu

} // namespace migraph
