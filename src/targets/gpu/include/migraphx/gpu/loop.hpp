#ifndef MIGRAPHX_GUARD_RTGLIB_LOOP_HPP
#define MIGRAPHX_GUARD_RTGLIB_LOOP_HPP

#include <migraphx/argument.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/op/loop.hpp>
#include <migraphx/gpu/miopen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_loop
{
    op::loop op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::loop"; }
    shape compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const;
    argument
    compute(const shape& output_shape,
            const std::vector<argument>& args,
            const std::vector<module_ref>& mods,
            const std::function<std::vector<argument>(
                module_ref&, const std::unordered_map<std::string, argument>&)>& run) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
