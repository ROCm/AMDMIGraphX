#ifndef MIGRAPHX_GUARD_RTGLIB_CONCAT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONCAT_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/concat.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_concat
{
    op::concat op;

    std::string name() const { return "gpu::concat"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    int output_alias(const std::vector<shape>& shapes) const { return shapes.size() - 1; }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
