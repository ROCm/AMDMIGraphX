#include <migraphx/gpu/shape_of.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_shape::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2);
    return {shape::int64_type, {1, inputs[0].lens().size()}};
}

argument hip_shape::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::shape_of(ctx.get_stream().get(), args[1], args[0]);
    return args[1];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
