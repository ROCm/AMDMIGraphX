#ifndef MIGRAPHX_GUARD_RTGLIB_CONVERT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONVERT_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/convert.hpp>
#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/convert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_convert : unary_device<hip_convert, device::convert>
{
    op::convert op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    hip_convert(op::convert oper) : op(oper) {}

    shape compute_shape(std::vector<shape> inputs) const
    {
        inputs.pop_back();
        check_shapes{inputs}.packed();
        return op.compute_shape(inputs);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
