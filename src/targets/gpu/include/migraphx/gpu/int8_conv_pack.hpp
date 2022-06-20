#ifndef MIGRAPHX_GUARD_RTGLIB_INT8_CONV_PACK_HPP
#define MIGRAPHX_GUARD_RTGLIB_INT8_CONV_PACK_HPP

#include <migraphx/op/quant_dot.hpp>
#include <migraphx/config.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_int8_conv_pack
{
    std::string name() const { return "gpu::int8_conv_pack"; }
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
