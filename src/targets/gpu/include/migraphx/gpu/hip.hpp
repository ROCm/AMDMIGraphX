#ifndef MIGRAPH_GUARD_MIGRAPHLIB_HIP_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_HIP_HPP

#include <migraphx/operators.hpp>
#include <migraphx/config.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

migraphx::argument allocate_gpu(const migraphx::shape& s, bool host = false);

migraphx::argument to_gpu(migraphx::argument arg, bool host = false);

migraphx::argument from_gpu(migraphx::argument arg);

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
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
