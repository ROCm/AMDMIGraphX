#ifndef MIGRAPH_GUARD_MIGRAPHLIB_HIP_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_HIP_HPP

#include <migraph/operators.hpp>

namespace migraph {
namespace gpu {

migraph::argument allocate_gpu(migraph::shape s);

migraph::argument to_gpu(migraph::argument arg);

migraph::argument from_gpu(migraph::argument arg);

void copy_to_gpu(char* dst, const char* src, std::size_t size);

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

struct hip_memcpy
{
    std::string name() const { return "hip_memcpy"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        return inputs.at(2);
    }
    argument compute(context&, shape output_shape, std::vector<argument> args) const {
        std::size_t * p_data = reinterpret_cast<std::size_t*>(args.at(1).data());
        char* dst = args.at(0).data() + p_data[0];
        const char* src = args.at(2).data();
        std::size_t size = args.at(2).get_shape().bytes();
        copy_to_gpu(dst, src, size);
        return {output_shape, dst};
    }
};        

} // namespace gpu

} // namespace migraph

#endif
