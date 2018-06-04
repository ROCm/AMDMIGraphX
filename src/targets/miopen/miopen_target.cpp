#include <rtg/miopen/miopen_target.hpp>
#include <rtg/manage_ptr.hpp>
#include <rtg/instruction.hpp>
#include <rtg/operators.hpp>

#include <miopen/miopen.h>

namespace rtg {
namespace miopen {


struct hip_allocate
{
    std::string name() const { return "hip::allocate"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.front();
    }
    argument compute(shape output_shape, std::vector<argument>) const
    {
        char * data = nullptr;
        // TODO: Check return status
        hipMalloc(&data, output_shape.bytes());
        return {output_shape, data};
    }
};

struct hip_free
{
    std::string name() const { return "hip::free"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return {};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        // TODO: Check return status
        hipFree(args.front().data());
        return {};
    }
};


using miopen_handle     = RTG_MANAGE_PTR(miopenHandle_t, miopenDestroy);
using tensor_descriptor = RTG_MANAGE_PTR(miopenTensorDescriptor_t, miopenDestroyTensorDescriptor);
using convolution_descriptor = RTG_MANAGE_PTR(miopenConvolutionDescriptor_t,
                                              miopenDestroyConvolutionDescriptor);
using activation_descriptor  = RTG_MANAGE_PTR(miopenActivationDescriptor_t,
                                             miopenDestroyActivationDescriptor);

template <class Result, class F, class... Ts>
Result make_obj(F f, Ts... xs)
{
    typename Result::pointer x = nullptr;
    auto status                = f(&x, xs...);
    Result r{x};
    if(status != miopenStatusSuccess)
        RTG_THROW("MIOpen call failed");
    return r;
}

tensor_descriptor make_tensor(const rtg::shape& s)
{
    auto t = make_obj<tensor_descriptor>(&miopenCreateTensorDescriptor);
    // Convert to ints
    std::vector<int> lens(s.lens().begin(), s.lens().end());
    std::vector<int> strides(s.strides().begin(), s.strides().end());
    miopenDataType_t d;
    if(s.type() == shape::float_type)
        d = miopenFloat;
    else
        RTG_THROW("Unsupported type");
    miopenSetTensorDescriptor(t.get(), d, s.lens().size(), lens.data(), strides.data());
    return t;
}

convolution_descriptor make_conv(const rtg::convolution& op)
{
    auto c = make_obj<convolution_descriptor>(&miopenCreateConvolutionDescriptor);
    miopenInitConvolutionDescriptor(c.get(),
                                    miopenConvolution,
                                    op.padding[0],
                                    op.padding[1],
                                    op.stride[0],
                                    op.stride[1],
                                    op.dilation[0],
                                    op.dilation[1]);
    return c;
}

activation_descriptor make_relu()
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    miopenSetActivationDescriptor(ad.get(), miopenActivationRELU, 0, 0, 0);
    return ad;
}

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
        miopenFindConvolutionForwardAlgorithm(args[0].get(),
                                              x_desc.get(),
                                              args[1].get(),
                                              w_desc.get(),
                                              args[2].get(),
                                              cd.get(),
                                              y_desc.get(),
                                              args[3].get(),
                                              1,
                                              &algo_count,
                                              &perf,
                                              nullptr,
                                              0,
                                              false);
        miopenConvolutionForward(args[0].get(),
                                 &alpha,
                                 x_desc.get(),
                                 args[1].get(),
                                 w_desc.get(),
                                 args[2].get(),
                                 cd.get(),
                                 perf.fwd_algo,
                                 &beta,
                                 y_desc.get(),
                                 args[3].get(),
                                 nullptr,
                                 0);
        return args[3];
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
        miopenActivationForward(args[0].get(), ad.get(), &alpha, x_desc.get(), args[1].get(), &beta, y_desc.get(), args[2].get());

        return args[2];
    }
};

struct miopen_apply
{
    program* prog;
    instruction_ref handle;

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
        }
    }

    instruction_ref insert_allocation(instruction_ref ins, const shape& s)
    {
        if (ins == --prog->end())
        {
            return prog->add_parameter("output", s);
        }
        else
        {
            auto is = prog->add_outline(s);
            auto result = prog->insert_instruction(ins, hip_allocate{}, is);
            prog->insert_instruction(++ins, hip_free{}, result);
            return result;
        }
    }

    void apply_convolution(instruction_ref ins)
    {
        auto&& op = any_cast<convolution>(ins->op);
        auto cd = make_conv(op);
        auto output = insert_allocation(ins, ins->result);

        prog->replace_instruction(ins, miopen_convolution{op, std::move(cd)}, handle, ins->arguments.at(0), ins->arguments.at(1), output);
    }

    void apply_activation(instruction_ref ins) 
    {
        auto&& op = any_cast<activation>(ins->op);
        auto ad = make_relu();
        if(op.mode == "relu") 
        {
            auto output = insert_allocation(ins, ins->result);
            prog->replace_instruction(ins, miopen_relu{std::move(ad)}, handle, ins->arguments.at(0), output);
        }
    }
};

std::string miopen_target::name() const { return "miopen"; }

void miopen_target::apply(program& p) const { miopen_apply{&p}.apply(); }

} // namespace miopen

} // namespace rtg
