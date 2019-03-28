#ifndef MIGRAPHX_GUARD_RTGLIB_GEMM_HPP
#define MIGRAPHX_GUARD_RTGLIB_GEMM_HPP

#include <migraphx/shape.hpp>
#include <migraphx/operators.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_gemm
{
    op::dot op;
    std::string name() const { return "gpu::gemm"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    int output_alias(const std::vector<shape>& shapes) const { return shapes.size() - 1; }

    private:
    argument
    batch_matmul(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
