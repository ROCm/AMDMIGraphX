#ifndef MIGRAPHX_GUARD_RTGLIB_WHERE_HPP
#define MIGRAPHX_GUARD_RTGLIB_WHERE_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/where.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_where : ternary_device<hip_where, device::where>
{
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(4).same_dims();
        auto s1 = inputs.at(1);
        auto s2 = inputs.at(2);
        if(s1 == s2 and s1.packed())
        {
            return s1;
        }
        else if(s1.packed() != s2.packed())
        {
            return s1.packed() ? s1 : s2;
        }
        else if(s1.broadcasted() != s2.broadcasted())
        {
            return s1.broadcasted() ? s2.with_lens(s1.lens()) : s1.with_lens(s1.lens());
        }
        else
        {
            return {s1.type(), s1.lens()};
        }
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
