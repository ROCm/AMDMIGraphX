#include <migraphx/gpu/mul.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

shape hip_mul::compute_shape(const std::vector<shape>& inputs) const
{
    // check_shapes{inputs, *this}.has(3).standard();
    check_shapes{inputs, *this}.has(3);
    return inputs.at(0);
}

argument hip_mul::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::mul(ctx.get_stream().get(), args[2], args[0], args[1]);
    return args[2];
}

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
