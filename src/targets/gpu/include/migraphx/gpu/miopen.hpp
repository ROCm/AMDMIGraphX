#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_MIOPEN_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_MIOPEN_HPP

#include <migraphx/manage_ptr.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/lrn.hpp>
#include <miopen/miopen.h>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using miopen_handle          = MIGRAPHX_MANAGE_PTR(miopenHandle_t, miopenDestroy);
using tensor_descriptor      = MIGRAPHX_MANAGE_PTR(miopenTensorDescriptor_t,
                                              miopenDestroyTensorDescriptor);
using convolution_descriptor = MIGRAPHX_MANAGE_PTR(miopenConvolutionDescriptor_t,
                                                   miopenDestroyConvolutionDescriptor);
using pooling_descriptor     = MIGRAPHX_MANAGE_PTR(miopenPoolingDescriptor_t,
                                               miopenDestroyPoolingDescriptor);
using activation_descriptor  = MIGRAPHX_MANAGE_PTR(miopenActivationDescriptor_t,
                                                  miopenDestroyActivationDescriptor);
using fusion_plan_descriptor = MIGRAPHX_MANAGE_PTR(miopenFusionPlanDescriptor_t,
                                                   miopenDestroyFusionPlan);
using fused_operator_args    = MIGRAPHX_MANAGE_PTR(miopenOperatorArgs_t, miopenDestroyOperatorArgs);

using lrn_descriptor = MIGRAPHX_MANAGE_PTR(miopenLRNDescriptor_t, miopenDestroyLRNDescriptor);

template <class Result, class F, class... Ts>
Result make_obj(F f, Ts... xs)
{
    typename Result::pointer x = nullptr;
    auto status                = f(&x, xs...);
    Result r{x};
    if(status != miopenStatusSuccess)
        MIGRAPHX_THROW("MAKE_OBJ: MIOpen call failed");
    return r;
}

inline tensor_descriptor make_tensor(const migraphx::shape& s, bool pack = false)
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
    else if(s.type() == shape::int32_type)
        d = miopenInt32;
    else if(s.type() == shape::int8_type)
    {
        if(pack)
        {
            // update the lens and corresponding strides
            d          = miopenInt8x4;
            lens[1]    = ((lens[1] + 3) / 4) * 4;
            strides[0] = strides[1] * lens[1];
        }
        else
        {
            d = miopenInt8;
        }
    }
    else
    {
        MIGRAPHX_THROW("MAKE_TENSOR: unsupported type");
    }
    miopenSetTensorDescriptor(t.get(), d, s.lens().size(), lens.data(), strides.data());

    return t;
}

template <class T>
inline convolution_descriptor make_conv(const T& op)
{
    auto c = make_obj<convolution_descriptor>(&miopenCreateConvolutionDescriptor);
    miopenConvolutionMode_t c_mode = miopenConvolution;
    if(op.group > 1)
        c_mode = miopenGroupConv;
    miopenInitConvolutionDescriptor(c.get(),
                                    c_mode,
                                    op.padding[0],
                                    op.padding[1],
                                    op.stride[0],
                                    op.stride[1],
                                    op.dilation[0],
                                    op.dilation[1]);
    if(op.group > 1)
        miopenSetConvolutionGroupCount(c.get(), op.group);
    return c;
}

inline pooling_descriptor make_pooling(const migraphx::op::pooling& op)
{
    miopenPoolingMode_t mode;
    if(op.mode == "max")
        mode = miopenPoolingMax;
    else if(op.mode == "average")
        mode = miopenPoolingAverage;
    else
        MIGRAPHX_THROW("Unknown mode for pooling: " + op.mode);
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

inline lrn_descriptor make_lrn(const migraphx::op::lrn& op)
{
    auto ldesc = make_obj<lrn_descriptor>(&miopenCreateLRNDescriptor);
    miopenSetLRNDescriptor(ldesc.get(), miopenLRNCrossChannel, op.size, op.alpha, op.beta, op.bias);
    return ldesc;
}

inline activation_descriptor make_relu()
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    miopenSetActivationDescriptor(ad.get(), miopenActivationRELU, 0, 0, 0);
    return ad;
}

inline activation_descriptor make_sigmoid()
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    miopenSetActivationDescriptor(ad.get(), miopenActivationLOGISTIC, 0, 0, 0);
    return ad;
}

inline activation_descriptor make_tanh()
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    // onnx operator does not apply additional scaling for tanh
    // defaults for alpha and beta are therefore set to 1
    miopenSetActivationDescriptor(ad.get(), miopenActivationTANH, 1, 1, 0);
    return ad;
}

inline activation_descriptor make_abs()
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    miopenSetActivationDescriptor(ad.get(), miopenActivationABS, 0, 0, 0);
    return ad;
}

inline activation_descriptor make_leaky_relu(double alpha)
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    miopenSetActivationDescriptor(ad.get(), miopenActivationLEAKYRELU, alpha, 0, 0);
    return ad;
}

inline activation_descriptor make_elu(double alpha)
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    miopenSetActivationDescriptor(ad.get(), miopenActivationELU, alpha, 0, 0);
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

template <class F>
auto reflect(miopenActivationDescriptor_t ad, F f)
{
    assert(ad != nullptr);
    miopenActivationMode_t mode = miopenActivationPASTHRU;
    double alpha                = 0.0;
    double beta                 = 0.0;
    double gamma                = 0.0;
    miopenGetActivationDescriptor(ad, &mode, &alpha, &beta, &gamma);
    return pack(f(std::move(mode), "mode"),    // NOLINT
                f(std::move(alpha), "alpha"),  // NOLINT
                f(std::move(beta), "beta"),    // NOLINT
                f(std::move(gamma), "gamma")); // NOLINT
}

template <class F>
auto reflect(miopenLRNDescriptor_t lrnd, F f)
{
    assert(lrnd != nullptr);
    miopenLRNMode_t mode = miopenLRNWithinChannel;
    unsigned int n       = 0;
    double alpha         = 0.0;
    double beta          = 0.0;
    double k             = 0.0;
    miopenGetLRNDescriptor(lrnd, &mode, &n, &alpha, &beta, &k);
    return pack(f(std::move(mode), "mode"),   // NOLINT
                f(std::move(n), "n"),         // NOLINT
                f(std::move(alpha), "alpha"), // NOLINT
                f(std::move(beta), "beta"),   // NOLINT
                f(std::move(k), "k"));        // NOLINT
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
