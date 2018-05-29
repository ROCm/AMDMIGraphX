#include <rtg/miopen/miopen_target.hpp>
#include <rtg/manage_ptr.hpp>

#include <miopen/miopen.h>

namespace rtg { namespace miopen {

using miopen_handle = RTG_MANAGE_PTR(miopenHandle_t, miopenDestroy);
using tensor_descriptor = RTG_MANAGE_PTR(miopenTensorDescriptor_t, miopenDestroyTensorDescriptor);
using convolution_descriptor = RTG_MANAGE_PTR(miopenConvolutionDescriptor_t, miopenDestroyConvolutionDescriptor);
using activation_descriptor = RTG_MANAGE_PTR(miopenActivationDescriptor_t, miopenDestroyActivationDescriptor);

struct miopen_apply
{
    program * prog;

    void apply()
    {
        for(auto it = prog->begin();it != prog->end();it++) {
            if (it->op.name() == "convolution") {
                apply_convolution(it);
            } else if (it->op.name() == "activation") {
                apply_activation(it);
            }
        }
    }

    void apply_convolution(instruction_ref ins)
    {
        // auto&& op = any_cast<convolution>(ins->op);
        // prog->replace_instruction(ins, miopen_convolution{op}, ins->arguments);
    }

    void apply_activation(instruction_ref ins)
    {

    }

};

std::string miopen_target::name() const
{
    return "miopen";
}

void miopen_target::apply(program& p) const
{
    miopen_apply{&p}.apply();
}

} // namespace miopen

} // namespace rtg
