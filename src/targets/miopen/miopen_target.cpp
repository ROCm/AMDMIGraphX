#include <rtg/miopen/miopen_target.hpp>
#include <rtg/manage_ptr.hpp>
#include <rtg/instruction.hpp>
#include <rtg/operators.hpp>
#include <rtg/miopen/miopen.hpp>
#include <rtg/miopen/hip.hpp>

namespace rtg {
namespace miopen {

struct miopen_convolution
{
    convolution op;
    shared<convolution_descriptor> cd;

    std::string name() const { return "miopen::convolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(4);
        return op.compute_shape({inputs.at(1), inputs.at(2)});
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        auto x_desc = make_tensor(args[1].get_shape());
        auto w_desc = make_tensor(args[2].get_shape());
        auto y_desc = make_tensor(output_shape);

        float alpha = 1, beta = 0;
        int algo_count;
        miopenConvAlgoPerf_t perf;
        miopenFindConvolutionForwardAlgorithm(args[0].implicit(),
                                              x_desc.get(),
                                              args[1].implicit(),
                                              w_desc.get(),
                                              args[2].implicit(),
                                              cd.get(),
                                              y_desc.get(),
                                              args[3].implicit(),
                                              1,
                                              &algo_count,
                                              &perf,
                                              nullptr,
                                              0,
                                              false);
        miopenConvolutionForward(args[0].implicit(),
                                 &alpha,
                                 x_desc.get(),
                                 args[1].implicit(),
                                 w_desc.get(),
                                 args[2].implicit(),
                                 cd.get(),
                                 perf.fwd_algo,
                                 &beta,
                                 y_desc.get(),
                                 args[3].implicit(),
                                 nullptr,
                                 0);
        return args[3];
    }
};

struct miopen_pooling
{
    pooling op;
    shared<pooling_descriptor> pd;

    std::string name() const { return "miopen::pooling"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(3);
        return op.compute_shape({inputs.at(1)});
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        auto x_desc = make_tensor(args[1].get_shape());
        auto y_desc = make_tensor(output_shape);

        float alpha = 1, beta = 0;

        miopenPoolingForward(args[0].implicit(),
                                                  pd.get(),
                                                  &alpha,
                                                  x_desc.get(),
                                              args[1].implicit(),
                                                  &beta,
                                                  y_desc.get(),
                                              args[2].implicit(),
                                                  false,
                                                  nullptr,
                                                  0);

        return args[2];
    }
};

struct miopen_relu
{
    shared<activation_descriptor> ad;
    std::string name() const { return "miopen::relu"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(3);
        return inputs.at(1);
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        float alpha = 1, beta = 0;
        auto x_desc = make_tensor(args[1].get_shape());
        auto y_desc = make_tensor(output_shape);
        miopenActivationForward(args[0].implicit(),
                                ad.get(),
                                &alpha,
                                x_desc.get(),
                                args[1].implicit(),
                                &beta,
                                y_desc.get(),
                                args[2].implicit());

        return args[2];
    }
};

struct miopen_apply
{
    program* prog = nullptr;
    instruction_ref handle{};

    void apply()
    {
        handle = prog->add_parameter("handle", shape{shape::any_type});
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
                                  handle,
                                  ins->arguments.at(0),
                                  ins->arguments.at(1),
                                  output);
    }

    void apply_pooling(instruction_ref ins)
    {
        auto&& op   = any_cast<pooling>(ins->op);
        auto pd     = make_pooling(op);
        auto output = insert_allocation(ins, ins->result);

        prog->replace_instruction(ins,
                                  miopen_pooling{op, std::move(pd)},
                                  handle,
                                  ins->arguments.at(0),
                                  output);
    }

    void apply_activation(instruction_ref ins)
    {
        auto&& op = any_cast<activation>(ins->op);
        auto ad   = make_relu();
        if(op.mode == "relu")
        {
            auto output = insert_allocation(ins, ins->result);
            prog->replace_instruction(
                ins, miopen_relu{std::move(ad)}, handle, ins->arguments.at(0), output);
        }
    }
};

std::string miopen_target::name() const { return "miopen"; }

void miopen_target::apply(program& p) const { miopen_apply{&p}.apply(); }

} // namespace miopen

} // namespace rtg
