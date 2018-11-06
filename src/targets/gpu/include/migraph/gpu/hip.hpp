#ifndef MIGRAPH_GUARD_MIGRAPHLIB_HIP_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_HIP_HPP

#include <migraph/operators.hpp>
#include <utility>

namespace migraph {
namespace gpu {

migraph::argument allocate_gpu(const migraph::shape& s, bool host = false);

migraph::argument to_gpu(migraph::argument arg, bool host = false);

migraph::argument from_gpu(migraph::argument arg);

void set_device(std::size_t id);

void gpu_sync();

void copy_to_gpu(argument src, argument dst);

struct hip_allocate
{
    std::string tag{};
    std::string name() const { return "hip::allocate"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.front();
    }
    argument compute(context&, const shape& output_shape, const std::vector<argument>&) const
    {
        return allocate_gpu(output_shape);
    }
};

struct hip_sync
{
    std::string tag{};
    std::string name() const { return "hip::sync"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        if(inputs.empty())
            return {};
        else
            return inputs.front();
    }
    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        gpu_sync();
        if(args.empty())
            return {};
        else
            return args.front();
    }
};

struct hip_write
{
    std::string name() const { return "hip::write"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.front();
    }
    argument compute(context&, const shape&, const std::vector<argument>& args) const
    {
        return to_gpu(args.front());
    }
    int output_alias(const std::vector<shape>&) const { return 0; }
};

struct hip_copy
{
    std::string name() const { return "hip_copy"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(2);
        return inputs.at(1);
    }
    argument compute(context&, const shape&, std::vector<argument> args) const
    {
        copy_to_gpu(args[0], args[1]);
        return args[1];
    }
    int output_alias(const std::vector<shape>&) const { return 1; }
};
} // namespace gpu
} // namespace migraph

#endif
