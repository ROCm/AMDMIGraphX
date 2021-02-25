#ifndef MIGRAPHX_GUARD_RTGLIB_IF_HPP
#define MIGRAPHX_GUARD_RTGLIB_IF_HPP

#include <migraphx/shape.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/op/if_op.hpp>
#include <migraphx/module_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_if
{
    op::if_op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::if"; }
    shape compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const;
    argument compute(
        const std::vector<argument>& args,
        const std::vector<module_ref>& mods,
        std::function<std::vector<argument>(module_ref& mdl, const std::vector<argument>& inputs)>
            run) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
