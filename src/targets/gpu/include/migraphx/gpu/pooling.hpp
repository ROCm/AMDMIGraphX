#ifndef MIGRAPHX_GUARD_RTGLIB_POOLING_HPP
#define MIGRAPHX_GUARD_RTGLIB_POOLING_HPP

#include <migraphx/argument.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/gpu/miopen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_pooling
{
    op::pooling op;
    shared<pooling_descriptor> pd;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::pooling"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    void finalize(context&, const shape&, const std::vector<shape>&);
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
