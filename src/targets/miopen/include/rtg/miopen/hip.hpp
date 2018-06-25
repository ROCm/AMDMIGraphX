#ifndef RTG_GUARD_RTGLIB_HIP_HPP
#define RTG_GUARD_RTGLIB_HIP_HPP

#include <rtg/manage_ptr.hpp>
#include <rtg/argument.hpp>
#include <rtg/operators.hpp>
#include <miopen/miopen.h>

#include <vector>

namespace rtg {
namespace miopen {

rtg::argument allocate_gpu(rtg::shape s);

rtg::argument to_gpu(rtg::argument arg);

rtg::argument from_gpu(rtg::argument arg);

struct hip_allocate
{
    std::string name() const { return "hip::allocate"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.front();
    }
    argument compute(shape output_shape, std::vector<argument>) const
    {
        return allocate_gpu(output_shape);
    }
};

} // namespace miopen

} // namespace rtg

#endif
