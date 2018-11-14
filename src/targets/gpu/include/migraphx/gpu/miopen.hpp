#ifndef MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_MIOPEN_HPP

#include <migraphx/manage_ptr.hpp>
#include <migraphx/operators.hpp>
#include <miopen/miopen.h>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

using miopen_handle          = MIGRAPH_MANAGE_PTR(miopenHandle_t, miopenDestroy);
using tensor_descriptor      = MIGRAPH_MANAGE_PTR(miopenTensorDescriptor_t,
                                             miopenDestroyTensorDescriptor);
using convolution_descriptor = MIGRAPH_MANAGE_PTR(miopenConvolutionDescriptor_t,
                                                  miopenDestroyConvolutionDescriptor);
using pooling_descriptor     = MIGRAPH_MANAGE_PTR(miopenPoolingDescriptor_t,
                                              miopenDestroyPoolingDescriptor);
using activation_descriptor  = MIGRAPH_MANAGE_PTR(miopenActivationDescriptor_t,
                                                 miopenDestroyActivationDescriptor);
using fusion_plan_descriptor = MIGRAPH_MANAGE_PTR(miopenFusionPlanDescriptor_t,
                                                  miopenDestroyFusionPlan);
using fused_operator_args    = MIGRAPH_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);

template <class Result, class F, class... Ts>
Result make_obj(F f, Ts... xs)
{
    typename Result::pointer x = nullptr;
    auto status                = f(&x, xs...);
    Result r{x};
    if(status != miopenStatusSuccess)
        MIGRAPH_THROW("MIOpen call failed");
    return r;
}

inline tensor_descriptor make_tensor(const migraphx::shape& s)
{
    auto t = make_obj<tensor_descriptor>(&miopenCreateTensorDescriptor);
    // Convert to ints
    std::vector<int> lens(s.lens().begin(), s.lens().end());
    std::vector<int> strides(s.strides().begin(), s.strides().end());
    miopenDataType_t d;
    if(s.type() == shape::float_type)
        d = miopenFloat;
    else if(s.type() == shape::half_type)
        d = miopenHalf;
    else
        MIGRAPH_THROW("Unsupported type");
    miopenSetTensorDescriptor(t.get(), d, s.lens().size(), lens.data(), strides.data());
    return t;
}

inline convolution_descriptor make_conv(const migraphx::op::convolution& op)
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

inline pooling_descriptor make_pooling(const migraphx::op::pooling& op)
{
    miopenPoolingMode_t mode;
    if(op.mode == "max")
        mode = miopenPoolingMax;
    else
        mode = miopenPoolingAverage;
    auto p = make_obj<pooling_descriptor>(&miopenCreatePoolingDescriptor);
    miopenSet2dPoolingDescriptor(p.get(),
                                 mode,
                                 op.lengths[0],
                                 op.lengths[1],
                                 op.padding[0],
                                 op.padding[1],
                                 op.stride[0],
                                 op.stride[1]);
    return p;
}

inline activation_descriptor make_relu()
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    miopenSetActivationDescriptor(ad.get(), miopenActivationRELU, 0, 0, 0);
    return ad;
}

inline activation_descriptor make_leaky_relu(double alpha)
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    miopenSetActivationDescriptor(ad.get(), miopenActivationLEAKYRELU, alpha, 0, 0);
    return ad;
}

inline fusion_plan_descriptor make_fusion_plan(const shape& input)
{
    auto t = make_tensor(input);
    return make_obj<fusion_plan_descriptor>(&miopenCreateFusionPlan, miopenVerticalFusion, t.get());
}

// Temporary hack to workaround memory problems in miopen
inline fusion_plan_descriptor make_fusion_plan(const tensor_descriptor& input)
{
    return make_obj<fusion_plan_descriptor>(
        &miopenCreateFusionPlan, miopenVerticalFusion, input.get());
}

inline fused_operator_args make_fused_args()
{
    return make_obj<fused_operator_args>(&miopenCreateOperatorArgs);
}

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
