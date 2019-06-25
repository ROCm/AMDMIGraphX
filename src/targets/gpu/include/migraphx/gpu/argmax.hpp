#ifndef MIGRAPHX_GUARD_RTGLIB_ARGMAX_HPP
#define MIGRAPHX_GUARD_RTGLIB_ARGMAX_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/argmax.hpp>
#include <migraphx/gpu/device/argmax.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_argmax
{
    op::argmax op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::argmax"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
