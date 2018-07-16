#include <migraph/miopen/miopen_target.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/shape_for_each.hpp>
#include <migraph/miopen/miopen.hpp>
#include <migraph/miopen/hip.hpp>
#include <migraph/dfor.hpp>

namespace migraph {
namespace miopen {

struct miopen_context
{
    shared<miopen_handle> handle;
};

struct miopen_convolution
{
    convolution op;
    shared<convolution_descriptor> cd;

    std::string name() const { return "miopen::convolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context& gctx, shape output_shape, std::vector<argument> args) const
    {
        auto& ctx   = any_cast<miopen_context>(gctx);
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

    std::string name() const { return "miopen::pooling"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return op.compute_shape({inputs.at(1)});
    }
    argument compute(context& gctx, shape output_shape, std::vector<argument> args) const
    {
        auto& ctx   = any_cast<miopen_context>(gctx);
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
    std::string name() const { return "miopen::add"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return inputs.at(0);
    }

    argument compute(context& gctx, shape output_shape, std::vector<argument> args) const
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
            auto& ctx   = any_cast<miopen_context>(gctx);
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
    std::string name() const { return "miopen::convolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return op.compute_shape({inputs.at(0), inputs.at(1)});
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const
    {
        // visit_all(result, from_gpu(args[0]), from_gpu(args[1]))(
        //     [&](auto output, auto input1, auto input2) {
        //         dfor(input1.get_shape().lens()[0],
        //              input2.get_shape().lens()[1],
        //              input2.get_shape().lens()[0])(
        //             [&](auto i, auto j, auto k) { output(i, j) += input1(i, k) * input2(k, j); });
        //     });
        visit_all(args[2], args[0], args[1])(
            [&](auto output, auto input1, auto input2) {
                float alpha = 1.0;
                float beta = 0.0;
                rocblas_int lda = input1.get_shape().lens()[1];
                rocblas_int ldb = input2.get_shape().lens()[1];
                rocblas_int ldc = output.get_shape().lens()[1];
                rocblas_int m = ouptut.get_shape().lens()[0];
                rocblas_int n = ouptut.get_shape().lens()[1];
                rocblas_int k = args[0].get_shape().lens()[1];
                rocblas_sgemm(rochandle, rocblas_operation_none, rocblas_operation_none, n, m, k,
                                     &alpha,
                                     input2.data(), ldb,
                                     input1.data(), lda,
                                     &beta,
                                     output.data(), ldc);
            });
        return args[2];
    }
};

struct miopen_relu
{
    shared<activation_descriptor> ad;
    std::string name() const { return "miopen::relu"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        return inputs.at(1);
    }

    argument compute(context& gctx, shape output_shape, std::vector<argument> args) const
    {
        auto& ctx   = any_cast<miopen_context>(gctx);
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
        prog->insert_instruction(prog->begin(), check_context<miopen_context>{});
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
};

struct miopen_pass
{
    std::string name() const { return "miopen::pass"; }

    void apply(program& p) const { miopen_apply{&p}.apply(); }
};

std::vector<pass> miopen_target::get_passes(context&) const { return {miopen_pass{}}; }

std::string miopen_target::name() const { return "miopen"; }

context miopen_target::get_context() const
{
    return miopen_context{share(make_obj<miopen_handle>(&miopenCreate))};
}

} // namespace miopen

} // namespace migraph
