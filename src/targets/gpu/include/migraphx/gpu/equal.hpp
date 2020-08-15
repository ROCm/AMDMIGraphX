#ifndef MIGRAPHX_GUARD_RTGLIB_EQUAL_HPP
#define MIGRAPHX_GUARD_RTGLIB_EQUAL_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/equal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_equal : binary_device<hip_equal, device::equal>
{
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(3).standard();
        auto s0 = inputs.at(0);
        auto s1 = inputs.at(1);
        if(s0 == s1 and s0.packed())
        {
            return {shape::bool_type, s0.lens(), s0.strides()};
        }
        else
        {
            return {shape::bool_type, s0.lens()};
        }
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
