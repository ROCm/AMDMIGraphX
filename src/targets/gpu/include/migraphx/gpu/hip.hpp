#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_HIP_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_HIP_HPP

#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/check_shapes.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

argument allocate_gpu(const shape& s, bool host = false);

argument register_on_gpu(const argument& arg);

argument to_gpu(const argument& arg, bool host = false);

argument from_gpu(const argument& arg);

void set_device(std::size_t id);

void gpu_sync();
void stream_sync(context& ctx);

void gpu_copy(context& ctx, const argument& src, const argument& dst);
void copy_to_gpu(context& ctx, const argument& src, const argument& dst);
void copy_from_gpu(context& ctx, const argument& src, const argument& dst);

argument get_preallocation(context& ctx, const std::string& id);

struct hip_allocate
{
    shape s;
    std::string tag{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.tag, "tag"));
    }

    std::string name() const { return "hip::allocate"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs}.has(0);
        return s;
    }
    argument compute(context&, const shape& output_shape, const std::vector<argument>&) const
    {
        return allocate_gpu(output_shape);
    }
};

struct hip_sync_device
{
    std::string tag{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.tag, "tag"));
    }

    std::string name() const { return "hip::sync_device"; }
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

struct hip_copy_to_gpu
{
    std::string name() const { return "hip::copy_to_gpu"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1, 2);
        return inputs.at(0);
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        auto input = register_on_gpu(args[0]);
        if(args.size() == 1)
            return input;
        argument result = args[1].share();
        gpu_copy(ctx, input, result);
        // Associate the input since it was registered with hip
        return {result.get_shape(), [input, result]() mutable { return result.data(); }};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>& args) const
    {
        if(args.size() == 1)
            return -1;
        return 1;
    }
};

struct hip_copy_from_gpu
{
    std::string name() const { return "hip::copy_from_gpu"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1, 2);
        return inputs.at(0);
    }
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const
    {
        if(args.size() == 1)
        {
            argument result = allocate_gpu(output_shape, true);
            gpu_copy(ctx, args[0], result);
            return result;
        }
        copy_from_gpu(ctx, args[0], args[1]);

        return args[1];
    }
    std::ptrdiff_t output_alias(const std::vector<shape>& args) const
    {
        if(args.size() == 1)
            return -1;
        return 1;
    }
};

struct hip_copy
{
    std::string name() const { return "hip::copy"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(2).standard();
        return inputs.at(1);
    }
    argument compute(context& ctx, const shape&, std::vector<argument> args) const
    {
        gpu_copy(ctx, args[0], args[1]);
        return args[1];
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 1; }
};

struct hip_load_memory
{
    shape s;
    std::string id{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.s, "shape"), f(self.id, "id"));
    }

    std::string name() const { return "hip::hip_load_memory"; }
    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs}.has(0);
        return s;
    }
    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        return get_preallocation(ctx, id);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
