#ifndef RTG_GUARD_RTGLIB_MIOPEN_HPP
#define RTG_GUARD_RTGLIB_MIOPEN_HPP

#include <rtg/manage_ptr.hpp>

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

inline tensor_descriptor make_tensor(const rtg::shape& s)
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

inline convolution_descriptor make_conv(const rtg::convolution& op)
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

inline activation_descriptor make_relu()
{
    auto ad = make_obj<activation_descriptor>(&miopenCreateActivationDescriptor);
    miopenSetActivationDescriptor(ad.get(), miopenActivationRELU, 0, 0, 0);
    return ad;
}

} // namespace miopen

} // namespace rtg

#endif
