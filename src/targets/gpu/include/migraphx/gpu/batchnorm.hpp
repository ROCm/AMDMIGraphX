#ifndef MIGRAPHX_GUARD_RTGLIB_BATCHNORM_HPP
#define MIGRAPHX_GUARD_RTGLIB_BATCHNORM_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/batch_norm.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_batch_norm_inference
{
    op::batch_norm_inference op;
    std::string name() const { return "gpu::batch_norm_inference"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
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
