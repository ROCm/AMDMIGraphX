#include <migraph/gpu/concat.hpp>
#include <migraph/operators.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/device/concat.hpp>
#include <utility>

namespace migraph {
namespace gpu {

shape hip_concat::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

std::vector<std::size_t> hip_concat::compute_offsets(const shape& output_shape,
                                                     const std::vector<argument> args) const
{
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> offset(args[0].get_shape().lens().size(), 0);
    offset[op.axis] = 0;
    for(const auto& arg : args)
    {
        offsets.push_back(output_shape.index(offset));
        offset[op.axis] += arg.get_shape().lens()[op.axis];
    }
    return offsets;
}

argument hip_concat::compute(context& ctx,
                             const shape& output_shape,
                             const std::vector<argument>& args) const
{
    std::vector<std::size_t> offsets = compute_offsets(output_shape, args);
    return device::concat(output_shape, args, offsets);
}

} // namespace gpu

} // namespace migraph
