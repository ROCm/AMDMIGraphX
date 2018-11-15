#include <migraph/gpu/sin.hpp>
#include <migraph/operators.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/config.hpp>
#include <migraph/gpu/miopen.hpp>
#include <utility>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

shape hip_sin::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2);
    return inputs.at(0);
}

argument hip_sin::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    device::sin(ctx.get_stream().get(), args[1], args[0]);
    return args[1];
}

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraph
