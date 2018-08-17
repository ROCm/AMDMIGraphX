#ifndef MIGRAPH_GUARD_MIGRAPHLIB_HIP_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_HIP_HPP

#include <migraph/operators.hpp>

namespace migraph {
namespace gpu {

migraph::argument allocate_gpu(migraph::shape s, bool host=false);

migraph::argument to_gpu(migraph::argument arg, bool host=false);

migraph::argument from_gpu(migraph::argument arg);

struct hip_allocate
{
    std::string name() const { return "hip::allocate"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.front();
    }
    argument compute(context&, shape output_shape, std::vector<argument>) const
    {
        return allocate_gpu(output_shape);
    }
};

struct hip_write
{
    std::string name() const { return "hip::write"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.front();
    }
    argument compute(context&, shape, std::vector<argument> args) const
    {
        return to_gpu(args.front());
    }
};

} // namespace gpu

} // namespace migraph

#endif
