#ifndef MIGRAPHX_GUARD_RTGLIB_QUANT_GEMM_HPP
#define MIGRAPHX_GUARD_RTGLIB_QUANT_GEMM_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/quant_dot.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_quant_gemm
{
    op::quant_dot op;
    argument arg_a;
    argument arg_b;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::quant_gemm"; }
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
