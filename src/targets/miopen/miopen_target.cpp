#include <rtg/miopen/miopen_target.hpp>
#include <rtg/manage_ptr.hpp>
#include <rtg/instruction.hpp>
#include <rtg/operators.hpp>

#include <miopen/miopen.h>

namespace rtg {
namespace miopen {

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

struct miopen_convolution
{
    convolution op;
    convolution_descriptor cd;

    std::string name() const { return "miopen::convolution"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        return op.compute_shape({inputs.at(1), inputs.at(2)});
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        argument result;
        auto x_desc = make_tensor(args[1].get_shape());
        auto w_desc = make_tensor(args[2].get_shape());
        auto y_desc = make_tensor(output_shape);

        int algo_count;
        miopenConvAlgoPerf_t perf;
        miopenFindConvolutionForwardAlgorithm(args[0].data(),
                                              x_desc.get(),
                                              args[1].data(),
                                              w_desc,
                                              args[2].data(),
                                              cd.get(),
                                              y_desc,
                                              args[4].data(),
                                              1,
                                              &algo_count,
                                              &perf,
                                              args[3].data(),
                                              args[3].get_shape().bytes(),
                                              false);
        miopenConvolutionForward(args[0].data(),
                                 &alpha,
                                 x_desc,
                                 args[1].data(),
                                 w_desc,
                                 args[2].data(),
                                 cd.get(),
                                 perf.fwd_algo,
                                 &beta,
                                 y_desc,
                                 args[4].data(),
                                 args[3].data(),
                                 args[3].get_shape().bytes());
        return result;
    }
};

struct miopen_apply
{
    program* prog;

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
        }
    }

    void apply_convolution(instruction_ref ins)
    {
        // auto&& op = any_cast<convolution>(ins->op);
        // prog->replace_instruction(ins, miopen_convolution{op}, ins->arguments);
    }

    void apply_activation(instruction_ref ins) {}
};

std::string miopen_target::name() const { return "miopen"; }

void miopen_target::apply(program& p) const { miopen_apply{&p}.apply(); }

} // namespace miopen

} // namespace rtg
