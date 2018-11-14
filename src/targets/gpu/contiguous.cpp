#include <migraphx/gpu/contiguous.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

shape miopen_contiguous::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2);
    return op.compute_shape({inputs.at(0)});
}
argument miopen_contiguous::compute(context& ctx,
                                    shape output_shape,
                                    const std::vector<argument>& args) const
{
    assert(output_shape == args[1].get_shape());
    assert(output_shape.standard());
    (void)output_shape;
    device::contiguous(ctx.get_stream().get(), args.at(1), args.at(0));
    return args.at(1);
}

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
