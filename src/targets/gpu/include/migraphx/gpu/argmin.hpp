#ifndef MIGRAPHX_GUARD_RTGLIB_ARGMIN_HPP
#define MIGRAPHX_GUARD_RTGLIB_ARGMIN_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/argmin.hpp>
#include <migraphx/gpu/device/argmin.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_argmin
{
    op::argmin op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::argmin"; }
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
