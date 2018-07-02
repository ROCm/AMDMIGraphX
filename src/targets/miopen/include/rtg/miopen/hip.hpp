#ifndef RTG_GUARD_RTGLIB_HIP_HPP
#define RTG_GUARD_RTGLIB_HIP_HPP

#include <rtg/operators.hpp>

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
    argument compute(context&, shape output_shape, std::vector<argument>) const
    {
        return allocate_gpu(output_shape);
    }
};

} // namespace miopen

} // namespace rtg

#endif
