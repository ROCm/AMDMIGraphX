#ifndef MIGRAPHX_GUARD_RTGLIB_CLIP_HPP
#define MIGRAPHX_GUARD_RTGLIB_CLIP_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/clip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_clip
{
    op::clip op;
    std::string name() const { return "gpu::clip"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
