#include "migraphx/gpu/hip.hpp"
#include <iterator>
#include <migraphx/run_loop.hpp>
#include <migraphx/gpu/loop.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/fill.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_loop::compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
{
    auto input_num = (inputs.size() - 2) / 2;
    inputs.erase(inputs.begin() + input_num, inputs.end());
    return op.compute_shape(inputs, std::move(mods));
}

struct gpu_loop
{
    int64_t max_iter_num = 0;

    template <class T>
    void copy(context& ctx, const argument& src, T& dst) const
    {
        argument arg_dst{src.get_shape(), &dst};
        copy_from_gpu(ctx, src, arg_dst);
    }

    template <class T>
    void copy(context& ctx, T src, const argument& dst) const
    {
        argument arg_src{dst.get_shape(), &src};
        copy_to_gpu(ctx, arg_src, dst);
    }

    template <class InputIt, class OutputIt>
    void copy_carry_dependencies(context& ctx, InputIt first, InputIt last, OutputIt d_first) const
    {
        while(first != last)
        {
            gpu_copy(ctx, *first++, *d_first++);
        }
    }

    void append(const std::vector<argument>&, const std::vector<argument>&, const int) const {}

    void
    set_zero(context& ctx, const std::vector<argument>& concatenated_outputs, const int iter) const
    {
        if(iter >= max_iter_num)
            return;

        auto elem_num = max_iter_num - iter;
        for(const auto& out : concatenated_outputs)
        {
            auto s    = out.get_shape();
            auto size = s.bytes() / max_iter_num;
            auto lens = s.lens();
            lens[0]   = elem_num;
            shape ss{s.type(), lens};
            assert(ss.bytes() + iter * size <= out.get_shape().bytes());
            device::fill(ctx.get_stream().get(), argument::load(ss, out.data() + iter * size), 0);
        }
        ctx.finish();
    }

    template <class T>
    void print_params(const T& params) const
    {
        for(auto na : params)
        {
            std::cout << "gpu, name = " << na.first
                      << ", val = " << migraphx::gpu::from_gpu(na.second) << std::endl;
        }
    }

    template <class T>
    void print_outputs(const T& outputs) const
    {
        for(auto na : outputs)
        {
            std::cout << "gpu, output = " << migraphx::gpu::from_gpu(na) << std::endl;
        }
    }
};

argument
hip_loop::compute(context& ctx,
                  const shape&,
                  const std::vector<argument>& args,
                  const std::vector<module_ref>& mods,
                  const std::function<std::vector<argument>(
                      module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
{
    return run_loop(gpu_loop{op.max_iter_num}, ctx, args, mods, run);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
