#ifndef MIGRAPHX_GUARD_RTGLIB_SOFTMAX_HPP
#define MIGRAPHX_GUARD_RTGLIB_SOFTMAX_HPP

#include <migraphx/shape.hpp>
#include <migraphx/operators.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_softmax
{
    op::softmax op;
    std::string name() const { return "gpu::softmax"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    int output_alias(const std::vector<shape>& shapes) const { return shapes.size() - 1; }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
