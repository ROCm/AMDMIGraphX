#ifndef MIGRAPHX_GUARD_RTGLIB_CONTIGUOUS_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTIGUOUS_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/contiguous.hpp>
#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/contiguous.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_contiguous : unary_device<miopen_contiguous, &device::contiguous>
{
    std::string name() const { return "gpu::contiguous"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(2);
        if(inputs.front().standard())
            return inputs.front();
        auto lens = inputs.at(0).lens();
        auto t    = inputs.at(0).type();
        return {t, lens};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
