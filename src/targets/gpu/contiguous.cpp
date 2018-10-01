#include <migraph/gpu/contiguous.hpp>
#include <migraph/operators.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/gpu/miopen.hpp>
#include <utility>

namespace migraph {
namespace gpu {

shape miopen_contiguous::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2);
    return op.compute_shape({inputs.at(0)});
}
argument
miopen_contiguous::compute(context&, shape output_shape, const std::vector<argument>& args) const
{
    assert(output_shape == args[1].get_shape());
    assert(output_shape.standard());
    (void)output_shape;
    device::contiguous(args.at(1), args.at(0));
    return args.at(1);
}

} // namespace gpu

} // namespace migraph
