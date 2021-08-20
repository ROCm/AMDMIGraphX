#ifndef MIGRAPHX_GUARD_RTGLIB_WHERE_HPP
#define MIGRAPHX_GUARD_RTGLIB_WHERE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/op/where.hpp>
#include <migraphx/gpu/miopen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_where
{
    op::where op;

    std::string name() const { return "gpu::where"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
